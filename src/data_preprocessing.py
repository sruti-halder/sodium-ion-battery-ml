import pandas as pd

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    df = df.groupby("formula", as_index=False).agg({
        "voltage": "mean",
        "capacity": "mean",
        "volume_change": "mean",
        "Battery_ID": "first",
        "working_ion": "first",
        "formula_charge": "first",
        "formula_discharge": "first"
    })

    
    df = df.dropna(subset=["voltage", "capacity", "volume_change"])

    df = df[(df["voltage"] > 0) & (df["voltage"] < 5)]

    df = df[df["capacity"] > 0]
    df = df[df["capacity"] < df["capacity"].quantile(0.99)]

    df = df[df["volume_change"] >= 0]
    df = df[df["volume_change"] < df["volume_change"].quantile(0.99)]

    df = df.reset_index(drop=True)

    print("Cleaned shape:", df.shape)

    df.to_csv(output_path, index=False)
