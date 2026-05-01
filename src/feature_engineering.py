import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from matminer.featurizers.composition import ElementProperty


# MAGPIE FEATURES

def generate_magpie_features(input_path, output_path):
    df = pd.read_csv(input_path)

    df["composition"] = df["formula_discharge"].apply(lambda x: Composition(x))

    ep_feat = ElementProperty.from_preset("magpie")

    df_magpie = ep_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    df_magpie = df_magpie.dropna()

    df_magpie.to_csv(output_path, index=False)

    print("Magpie features shape:", df_magpie.shape)


# DOMAIN FEATURES

def extract_domain_features(formula):
    comp = Composition(formula)

    elements = comp.elements
    fractions = comp.fractional_composition.get_el_amt_dict()

    num_elements = len(elements)
    total_atoms = comp.num_atoms

    avg_atomic_weight = sum(el.atomic_mass * fractions[el.symbol] for el in elements)
    avg_en = sum(el.X * fractions[el.symbol] for el in elements if el.X is not None)
    avg_radius = sum(el.atomic_radius * fractions[el.symbol] for el in elements if el.atomic_radius is not None)

    na_fraction = fractions.get("Na", 0)

    transition_metals = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
    has_tm = int(any(el.symbol in transition_metals for el in elements))

    return pd.Series([
        num_elements,
        total_atoms,
        avg_atomic_weight,
        avg_en,
        avg_radius,
        na_fraction,
        has_tm
    ])


def generate_domain_features(input_path, output_path):
    df = pd.read_csv(input_path)

    feature_names = [
        "num_elements",
        "total_atoms",
        "avg_atomic_weight",
        "avg_electronegativity",
        "avg_atomic_radius",
        "na_fraction",
        "has_transition_metal"
    ]

    df[feature_names] = df["formula_discharge"].apply(extract_domain_features)

    df = df.dropna()
    df.to_csv(output_path, index=False)

    print("Domain features shape:", df.shape)


# STRUCTURAL FEATURES

def extract_mp_id(battery_id):
    return battery_id.split("_")[0]


def generate_structural_features(input_path, output_path, api_key):
    from mp_api.client import MPRester

    df = pd.read_csv(input_path)

    df["mp_id"] = df["Battery_ID"].apply(extract_mp_id)

    mp_ids = df["mp_id"].unique().tolist()
    print("Unique MP IDs:", len(mp_ids))

    mpr = MPRester(api_key)

    docs = mpr.summary.search(
        material_ids=mp_ids,
        fields=["material_id", "density", "volume", "nsites", "symmetry"]
    )

    structure_data = []

    for doc in docs:
        structure_data.append({
            "mp_id": str(doc.material_id),
            "density": doc.density,
            "volume": doc.volume,
            "nsites": doc.nsites,
            "spacegroup": doc.symmetry.number if doc.symmetry else None,
            "crystal_system": doc.symmetry.crystal_system if doc.symmetry else None
        })

    df_struct_features = pd.DataFrame(structure_data)

    df = df.merge(df_struct_features, on="mp_id", how="left")

    df["volume_per_atom"] = df["volume"] / df["nsites"]

    df = pd.get_dummies(df, columns=["crystal_system"], dtype=int)

    df = df.dropna()

    df.to_csv(output_path, index=False)

    print("Structural features shape:", df.shape)
