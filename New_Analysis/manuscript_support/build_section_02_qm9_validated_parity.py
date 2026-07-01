from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "manuscript" / "figures"
TABLE_DIR = ROOT / "manuscript" / "tables"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    sse = float(np.sum(err * err))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": len(y_true),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": float(1.0 - sse / sst),
    }


def load_dft_ref() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    path = (
        ROOT
        / "New_Analysis/qm9_dft_ref_boe_transfer/outputs/requested_parity_against_DFT_ref/qm9_requested_parity_predictions_against_DFT_ref.csv"
    )
    df = pd.read_csv(path)
    df = df.loc[df["model_key"] == "gemnet_DFT"].copy()
    y_true = df["y_true"].to_numpy(float)
    y_pred = df["y_pred_mean_shift"].to_numpy(float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[finite], y_pred[finite], metrics(y_true[finite], y_pred[finite])


def load_dft_ref_boe() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    path = ROOT / "New_Analysis/qm9_dft_ref_boe_transfer/outputs/qm9_DFT_ref_BoE_gemnet_predictions.csv"
    df = pd.read_csv(path)
    df = df.loc[df["model_id"] == "gemnet_DFT_ref_BoE"].copy()
    y_true = df["y_true"].to_numpy(float)
    y_pred = df["y_pred_mean_shift"].to_numpy(float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[finite], y_pred[finite], metrics(y_true[finite], y_pred[finite])


def deterministic_sample(y_true: np.ndarray, y_pred: np.ndarray, max_points: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    if len(y_true) <= max_points:
        return y_true, y_pred
    rng = np.random.default_rng(101)
    idx = rng.choice(len(y_true), size=max_points, replace=False)
    return y_true[idx], y_pred[idx]


def plot_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stats: dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
    visual_limits: tuple[float, float],
) -> None:
    x, y = deterministic_sample(y_true, y_pred)
    # Match the validated older parity plots. Metrics are still computed on all
    # finite molecules; these limits only define the visual window.
    lo, hi = visual_limits
    ax.scatter(x, y, s=2.0, color="#9fbfda", alpha=0.55, linewidths=0, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="#1f2a35", linestyle="-", linewidth=1.1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(color="#dfe5eb", linewidth=0.7, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color("#c7d0d9")
        spine.set_linewidth(0.8)
    ax.text(
        0.04,
        0.96,
        "\n".join(
            [
                "mean-shift corrected",
                fr"$R^2$ = {stats['r2']:.3f}",
                fr"MAE = {stats['mae']:.4f} Ha",
                fr"RMSE = {stats['rmse']:.4f} Ha",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#d6dce2", "alpha": 0.94},
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    panels = [
        (
            *load_dft_ref(),
            r"QM9: GemNet($E_{\rm ref}$)",
            r"True $E_{\rm ref}$ (Ha)",
            r"Predicted $E_{\rm ref}$ (Ha)",
            (-0.10, 1.00),
        ),
        (
            *load_dft_ref_boe(),
            r"QM9: GemNet($E_{\rm ref,BoE}$)",
            r"True $E_{\rm ref,BoE}$ (Ha)",
            r"Predicted $E_{\rm ref,BoE}$ (Ha)",
            (0.15, 0.55),
        ),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.35), constrained_layout=True)
    rows = []
    for i, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        y_true, y_pred, stats, title, xlabel, ylabel, visual_limits = panel
        plot_panel(ax, y_true, y_pred, stats, title, xlabel, ylabel, visual_limits)
        ax.text(
            -0.13,
            1.08,
            chr(ord("a") + i),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )
        rows.append({"panel": chr(ord("a") + i), "title": title, **stats})

    out = FIG_DIR / "fig_qm9_gemnet_dftref_vs_dftrefboe_validated_parity.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE: {out}")

    metrics_out = TABLE_DIR / "fig_qm9_gemnet_dftref_vs_dftrefboe_validated_parity_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_out, index=False)
    print(f"WROTE: {metrics_out}")


if __name__ == "__main__":
    main()
