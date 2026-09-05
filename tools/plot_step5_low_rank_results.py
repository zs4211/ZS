"""Create compact, reproducible figures from the Step 5 CSV outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "artifacts" / "step5_low_rank" / "diagnostics"


def main() -> None:
    spectra = pd.read_csv(DIRECTORY / "mode_spectra.csv")
    first = spectra[spectra["index"] == 1].copy()
    modes = ["x", "y", "z", "time", "component"]
    windows = list(dict.fromkeys(first["window"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, representation in zip(axes, ("eulerian", "storm_relative")):
        subset = first[first["representation"] == representation]
        for mode in modes:
            values = subset[subset["mode"] == mode].set_index("window").loc[windows]
            axis.plot(windows, 100.0 * values["cumulative_energy"], marker="o", label=mode)
        axis.set_title(representation.replace("_", " ").title())
        axis.set_xlabel("Window")
        axis.grid(alpha=.25)
        axis.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Mode rank-1 energy (%)")
    axes[1].legend(frameon=False, ncol=1)
    fig.suptitle("Step 5 common-domain singular-value decay")
    fig.tight_layout()
    fig.savefig(DIRECTORY / "mode_rank1_energy.png", dpi=180)
    plt.close(fig)

    metrics = pd.read_csv(DIRECTORY / "rank_reconstruction_metrics.csv")
    metrics = metrics[
        (metrics["domain"] == "A_common_dense")
        & metrics["rank_name"].isin([
            "compact_temporal", "moderate_temporal",
            "compact_rt_full", "moderate_rt_full",
        ])
    ]
    fig, axis = plt.subplots(figsize=(8.2, 5.6))
    markers = {"eulerian": "o", "storm_relative": "^"}
    for representation, group in metrics.groupby("representation"):
        scatter = axis.scatter(
            group["compression_factor"], 100.0 * group["relative_frobenius_error"],
            c=100.0 * group["zeta_preservation"], cmap="viridis",
            vmin=50.0, vmax=105.0, marker=markers[representation],
            edgecolors="black", linewidths=.35, s=58, label=representation.replace("_", " "),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Compression factor (raw entries / Tucker parameters)")
    axis.set_ylabel("Relative Frobenius error (%)")
    axis.grid(alpha=.25)
    axis.legend(frameon=False)
    colour = fig.colorbar(scatter, ax=axis)
    colour.set_label("Peak vertical-vorticity preservation (%)")
    axis.set_title("Compression–error–vorticity trade-off on common dense domains")
    fig.tight_layout()
    fig.savefig(DIRECTORY / "compression_tradeoff.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
