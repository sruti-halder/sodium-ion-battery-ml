import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_distributions(file_path):
    df = pd.read_csv(file_path)

    configs = [
        ("voltage", "Average Voltage (V)", "(a)", "#4878CF"),
        ("capacity", "Gravimetric Capacity (mAh g$^{-1}$)", "(b)", "#C44E52"),
        ("volume_change", "Volume Change (%)", "(c)", "#4CAF50"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    for i, (ax, (col, xlabel, panel, color)) in enumerate(zip(axes, configs)):
        data = df[col].dropna().values

        ax.hist(data, bins=35, density=True, color=color, alpha=0.4)

        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 400)
        ax.plot(x, kde(x), color=color)

        mean_val = data.mean()
        median_val = np.median(data)

        ax.axvline(mean_val, linestyle="--")
        ax.axvline(median_val, linestyle=":")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(panel, loc="left")

    plt.savefig("fig_target_distributions.png", dpi=300)
    plt.show()