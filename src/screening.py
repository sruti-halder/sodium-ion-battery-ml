import pandas as pd
import numpy as np

from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from matminer.featurizers.composition import ElementProperty


# GENERATING CANDIDATES

def generate_candidates():
    tms = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]
    candidates = []

    # Layered
    for m in tms:
        candidates.append(f"Na{m}O2")

    # Mixed layered
    for i in range(len(tms)):
        for j in range(i+1, len(tms)):
            candidates.append(f"Na{tms[i]}0.5{tms[j]}0.5O2")

    # Phosphates
    for m in tms:
        candidates.append(f"Na{m}PO4")

    # Fluorophosphates
    for m in tms:
        candidates.append(f"Na2{m}PO4F")

    # NASICON
    for m in tms:
        candidates.append(f"Na3{m}2(PO4)3")

    candidates = list(set(candidates))

    print("Total candidates:", len(candidates))

    return pd.DataFrame({"formula": candidates})


# FEATURE GENERATION

def extract_domain_features(formula):
    comp = Composition(formula)
    fractions = comp.fractional_composition.get_el_amt_dict()

    num_elements = len(fractions)
    total_atoms = comp.num_atoms

    avg_atomic_weight = sum(
        Element(el).atomic_mass * fractions[el] for el in fractions
    )

    avg_en = sum(
        Element(el).X * fractions[el]
        for el in fractions if Element(el).X is not None
    )

    avg_radius = sum(
        Element(el).atomic_radius * fractions[el]
        for el in fractions if Element(el).atomic_radius is not None
    )

    na_fraction = fractions.get("Na", 0)

    transition_metals = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
    has_tm = int(any(el in transition_metals for el in fractions))

    return pd.Series([
        num_elements,
        total_atoms,
        avg_atomic_weight,
        avg_en,
        avg_radius,
        na_fraction,
        has_tm
    ])


def generate_features(candidate_df, X_train_columns):

    domain_cols = [
        "num_elements", "total_atoms", "avg_atomic_weight",
        "avg_electronegativity", "avg_atomic_radius",
        "na_fraction", "has_transition_metal"
    ]

    # Domain features
    candidate_df[domain_cols] = candidate_df["formula"].apply(extract_domain_features)

    # Magpie features
    candidate_df["composition"] = candidate_df["formula"].apply(lambda x: Composition(x))

    ep_feat = ElementProperty.from_preset("magpie")

    candidate_df = ep_feat.featurize_dataframe(
        candidate_df,
        col_id="composition",
        ignore_errors=True
    )

    # Match training columns
    missing_cols = set(X_train_columns) - set(candidate_df.columns)
    for col in missing_cols:
        candidate_df[col] = 0

    X_candidate = candidate_df[X_train_columns]

    return candidate_df, X_candidate


# SCREENING

def run_screening(candidate_df, X_candidate, model_voltage, model_capacity):

    candidate_df["pred_voltage"] = model_voltage.predict(X_candidate)
    candidate_df["pred_capacity"] = model_capacity.predict(X_candidate)

    # Score
    candidate_df["screening_score"] = (
        candidate_df["pred_voltage"] *
        candidate_df["pred_capacity"]
    )

    # Top materials
    top10 = candidate_df.sort_values(
        "screening_score",
        ascending=False
    ).head(10)

    print("\n=== TOP 10 MATERIALS ===")
    print(top10[[
        "formula",
        "pred_voltage",
        "pred_capacity",
        "screening_score"
    ]])

    return candidate_df



# CLASSIFICATION

def classify_materials(candidate_df):

    def classify(v):
        if v >= 2.5:
            return "Cathode"
        elif v <= 1.5:
            return "Anode"
        else:
            return "Intermediate"

    candidate_df["type"] = candidate_df["pred_voltage"].apply(classify)

    cathodes = candidate_df[candidate_df["type"] == "Cathode"]
    anodes = candidate_df[candidate_df["type"] == "Anode"]

    print("\n=== TOP CATHODES ===")
    print(cathodes.sort_values("pred_voltage", ascending=False).head(10)[
        ["formula", "pred_voltage"]
    ])

    print("\n=== TOP ANODES ===")
    print(anodes.sort_values("pred_voltage", ascending=True).head(10)[
        ["formula", "pred_voltage"]
    ])

    return candidate_df
