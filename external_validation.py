import pandas as pd
import numpy as np

from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from matminer.featurizers.composition import ElementProperty

from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# COMMON DOMAIN FEATURES

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

    transition_metals = [
        "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"
    ]

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


# MUKHERJEE VALIDATION (CAPACITY)

def validate_mukherjee(model, X_train_columns, file_path):
    df = pd.read_csv(file_path)
    df["formula"] = df["formula"].str.strip()

    domain_cols = [
        "num_elements", "total_atoms", "avg_atomic_weight",
        "avg_electronegativity", "avg_atomic_radius",
        "na_fraction", "has_transition_metal"
    ]

    df[domain_cols] = df["formula"].apply(extract_domain_features)

    df["composition"] = df["formula"].apply(lambda x: Composition(x))

    ep_feat = ElementProperty.from_preset("magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)

    # Match training columns
    missing_cols = set(X_train_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    X = df[X_train_columns]

    df["predicted_capacity"] = model.predict(X)

    print("\n=== MUKHERJEE VALIDATION ===")
    print(df[["formula", "predicted_capacity", "experimental_capacity"]])

    print("MAE:",
          mean_absolute_error(df["experimental_capacity"], df["predicted_capacity"]))
    print("R2:",
          r2_score(df["experimental_capacity"], df["predicted_capacity"]))


# JOSHI VALIDATION (VOLTAGE)

def validate_joshi(model_gb, model_xgb, X_train_columns, file_path):
    df = pd.read_csv(file_path)
    df["formula"] = df["formula"].str.strip()

    domain_cols = [
        "num_elements", "total_atoms", "avg_atomic_weight",
        "avg_electronegativity", "avg_atomic_radius",
        "na_fraction", "has_transition_metal"
    ]

    df[domain_cols] = df["formula"].apply(extract_domain_features)

    df["composition"] = df["formula"].apply(lambda x: Composition(x))

    ep_feat = ElementProperty.from_preset("magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition", ignore_errors=True)

    missing_cols = set(X_train_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    X = df[X_train_columns]

    df["voltage_gb"] = model_gb.predict(X)
    df["voltage_xgb"] = model_xgb.predict(X)

    print("\n=== JOSHI VALIDATION ===")
    print(df[["formula", "voltage_gb", "voltage_xgb", "Exp_volatge"]])

    plot_voltage_parity(df)


# PLOT

def plot_voltage_parity(df):
    y_true = df["Exp_volatge"]

    plt.figure(figsize=(5,4))
    plt.scatter(y_true, df["voltage_gb"], label="GB")
    plt.scatter(y_true, df["voltage_xgb"], label="XGB")

    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "--", color="black")

    plt.xlabel("Experimental Voltage")
    plt.ylabel("Predicted Voltage")
    plt.legend()
    plt.grid()
    plt.savefig("figures/voltage_parity.png")
    plt.show()