import pandas as pd

def show_statistics(file_path):
    df = pd.read_csv(file_path)

    cols = ["voltage", "capacity", "volume_change"]

    stats = df[cols].agg(["mean", "std", "min", "max"]).T
    stats.columns = ["Mean", "Std Dev", "Min", "Max"]

    stats = stats.round({
        "Mean": 2,
        "Std Dev": 2,
        "Min": 2,
        "Max": 2
    })

    print(stats)