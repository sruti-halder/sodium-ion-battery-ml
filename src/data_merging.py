import pandas as pd
import numpy as np

def create_hybrid_dataset(df_magpie, df_domain, df_struct):

    domain_cols = [
        "num_elements", "total_atoms", "avg_atomic_weight",
        "avg_electronegativity", "avg_atomic_radius",
        "na_fraction", "has_transition_metal"
    ]

    df = df_magpie.merge(
        df_domain[["formula"] + domain_cols],
        on="formula",
        how="inner"
    )

    df = df.merge(
        df_struct.drop(columns=["Battery_ID", "working_ion"]),
        on="formula",
        how="inner"
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df.select_dtypes(include=["number"]).drop(
        columns=["voltage", "capacity", "volume_change"]
    )

    y_v = df["voltage"]
    y_c = df["capacity"]
    y_vol = df["volume_change"]

    return X, y_v, y_c, y_vol
