from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "manuscript" / "figures"
TABLE_DIR = ROOT / "manuscript" / "tables"
INPUT_DIR = ROOT / "New_Analysis/manuscript_support/inputs"


@dataclass(frozen=True)
class ParityPanel:
    source_csv: Path
    title: str
    target_label: str
    true_col: str
    pred_col: str
    target_frame: str = "as_is"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    model_filter_col: str | None = None
    model_filter_value: str | None = None
    split_col: str | None = None
    split_value: str | None = None


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": float(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def load_panel(panel: ParityPanel) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not panel.source_csv.exists():
        raise FileNotFoundError(f"Missing prediction CSV for {panel.title}: {panel.source_csv}")
    df = pd.read_csv(panel.source_csv)
    if panel.model_filter_col is not None:
        df = df.loc[df[panel.model_filter_col].astype(str) == str(panel.model_filter_value)].copy()
    if panel.split_col is not None:
        df = df.loc[df[panel.split_col].astype(str) == str(panel.split_value)].copy()

    if panel.target_frame == "as_is":
        y_true = df[panel.true_col].to_numpy(float)
        y_pred = df[panel.pred_col].to_numpy(float)
    elif panel.target_frame == "tmqm_boe_to_full_ref":
        dft = pd.read_csv(ROOT / "New_Analysis/datasets/tmQM__DFT_ref-BoE.csv")
        dft["mol_id"] = dft["mol_id"].astype(str)
        df["mol_id"] = df["mol_id"].astype(str)
        df = df.merge(
            dft[["mol_id", "DFT_ref", "DFT_ref-BoE"]],
            on="mol_id",
            how="inner",
            validate="one_to_one",
        )
        baseline = df["DFT_ref"].to_numpy(float) - df["DFT_ref-BoE"].to_numpy(float)
        y_true = df["DFT_ref"].to_numpy(float)
        y_pred = baseline + df[panel.pred_col].to_numpy(float)
    elif panel.target_frame == "qm9_boe_to_full_ref":
        full_ref = pd.read_csv(
            ROOT
            / "New_Analysis/qm9_dft_ref_boe_transfer/outputs/requested_parity_against_DFT_ref/qm9_requested_parity_predictions_against_DFT_ref.csv"
        )
        full_ref = full_ref.loc[full_ref["model_key"].astype(str) == "gemnet_DFT", ["mol_id", "y_true"]].copy()
        full_ref = full_ref.rename(columns={"y_true": "DFT_ref"})
        df["mol_id"] = df["mol_id"].astype(str)
        full_ref["mol_id"] = full_ref["mol_id"].astype(str)
        df = df.merge(full_ref, on="mol_id", how="inner", validate="one_to_one")
        baseline = df["DFT_ref"].to_numpy(float) - df[panel.true_col].to_numpy(float)
        y_true = df["DFT_ref"].to_numpy(float)
        y_pred = baseline + df[panel.pred_col].to_numpy(float)
    else:
        raise ValueError(f"Unknown target_frame={panel.target_frame!r}")

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    return y_true, y_pred, metrics(y_true, y_pred)


def density_sample(y_true: np.ndarray, y_pred: np.ndarray, max_points: int = 25_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y_true) > max_points:
        rng = np.random.default_rng(101)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        x = y_true[idx]
        y = y_pred[idx]
    else:
        x = y_true
        y = y_pred
    values = np.vstack([x, y])
    try:
        z = gaussian_kde(values)(values)
    except Exception:
        z = np.ones_like(x)
    order = np.argsort(z)
    return x[order], y[order], z[order]


def plot_parity(ax: plt.Axes, panel: ParityPanel, y_true: np.ndarray, y_pred: np.ndarray, stats: dict[str, float]) -> None:
    x, y, z = density_sample(y_true, y_pred)
    if panel.xlim is not None:
        xlo, xhi = panel.xlim
    else:
        xlo = float(np.nanpercentile(np.r_[y_true, y_pred], 0.2))
        xhi = float(np.nanpercentile(np.r_[y_true, y_pred], 99.8))
        pad = 0.06 * (xhi - xlo)
        xlo -= pad
        xhi += pad

    if panel.ylim is not None:
        ylo, yhi = panel.ylim
    else:
        ylo = xlo
        yhi = xhi

    ax.scatter(x, y, c=z, s=5, cmap="viridis", alpha=0.72, linewidths=0, rasterized=True)
    diag_lo = min(xlo, ylo)
    diag_hi = max(xhi, yhi)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="#222222", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    if panel.xlim is None and panel.ylim is None:
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(panel.title, fontsize=10.5, pad=6)
    ax.set_xlabel(f"True {panel.target_label} (Ha)", fontsize=9.5)
    ax.set_ylabel(f"Predicted {panel.target_label} (Ha)", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(color="#d9d9d9", linewidth=0.6, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.text(
        0.04,
        0.96,
        "\n".join(
            [
                fr"$R^2$ = {stats['r2']:.3f}",
                fr"MAE = {stats['mae']:.4f} Ha",
                fr"RMSE = {stats['rmse']:.4f} Ha",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )


def panel_catalog(qm9_prediction: str) -> list[ParityPanel]:
    if qm9_prediction == "mean_shift":
        qm9_pred_col = "y_pred_mean_shift"
        qm9_title_prefix = "QM9 mean-shift"
    elif qm9_prediction == "raw":
        qm9_pred_col = "y_pred_raw"
        qm9_title_prefix = "QM9 raw transfer"
    else:
        raise ValueError(f"Unknown qm9_prediction={qm9_prediction!r}")

    return [
        ParityPanel(
            source_csv=INPUT_DIR / "tmqm_test_gemnet_dft_ref_predictions.csv",
            title=r"tmQM test: GemNet($E_{\rm ref}$)",
            target_label=r"$E_{\rm ref}$",
            true_col="y_true",
            pred_col="y_pred",
        ),
        ParityPanel(
            source_csv=INPUT_DIR / "tmqm_test_gemnet_dft_ref_boe_predictions.csv",
            title=r"tmQM test: BoE + GemNet($E_{\rm ref,BoE}$)",
            target_label=r"$E_{\rm ref}$",
            true_col="y_true",
            pred_col="y_pred",
            target_frame="tmqm_boe_to_full_ref",
        ),
        ParityPanel(
            source_csv=ROOT
            / "New_Analysis/qm9_dft_ref_boe_transfer/outputs/requested_parity_against_DFT_ref/qm9_requested_parity_predictions_against_DFT_ref.csv",
            title=rf"{qm9_title_prefix}: GemNet($E_{{\rm ref}}$)",
            target_label=r"$E_{\rm ref}$",
            true_col="y_true",
            pred_col=qm9_pred_col,
            xlim=(-0.30, 0.90) if qm9_prediction == "raw" else None,
            ylim=(-0.30, 2.25) if qm9_prediction == "raw" else None,
            model_filter_col="model_key",
            model_filter_value="gemnet_DFT",
        ),
        ParityPanel(
            source_csv=ROOT / "New_Analysis/qm9_dft_ref_boe_transfer/outputs/qm9_DFT_ref_BoE_gemnet_predictions.csv",
            title=rf"{qm9_title_prefix}: BoE + GemNet($E_{{\rm ref,BoE}}$)",
            target_label=r"$E_{\rm ref}$",
            true_col="y_true",
            pred_col=qm9_pred_col,
            target_frame="qm9_boe_to_full_ref",
            xlim=(-0.30, 0.90) if qm9_prediction == "raw" else None,
            ylim=(-0.30, 2.25) if qm9_prediction == "raw" else None,
            model_filter_col="model_id",
            model_filter_value="gemnet_DFT_ref_BoE",
        ),
    ]


def make_figure(qm9_prediction: str = "mean_shift", allow_missing_tmqm: bool = False) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    panels = panel_catalog(qm9_prediction)
    loaded: list[tuple[ParityPanel, np.ndarray, np.ndarray, dict[str, float]] | None] = []
    missing: list[Path] = []
    for panel in panels:
        try:
            y_true, y_pred, stats = load_panel(panel)
            loaded.append((panel, y_true, y_pred, stats))
        except FileNotFoundError:
            if allow_missing_tmqm and "tmqm_" in panel.source_csv.name:
                missing.append(panel.source_csv)
                loaded.append(None)
                continue
            raise

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.2), constrained_layout=True)
    row_labels = ["tmQM held-out test", "QM9 direct transfer"]
    panel_labels = ["a", "b", "c", "d"]
    summary_rows = []

    for ax, item, panel_label in zip(axes.ravel(), loaded, panel_labels, strict=True):
        ax.text(
            -0.14,
            1.09,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )
        if item is None:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "tmQM prediction CSV not found",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#555555",
            )
            continue
        panel, y_true, y_pred, stats = item
        plot_parity(ax, panel, y_true, y_pred, stats)
        summary_rows.append(
            {
                "panel": panel_label,
                "title": panel.title,
                "source_csv": str(panel.source_csv),
                "n": int(stats["n"]),
                "r2": stats["r2"],
                "mae": stats["mae"],
                "rmse": stats["rmse"],
            }
        )

    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.24,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    suffix = "" if qm9_prediction == "mean_shift" else "_qm9_raw"
    out = FIG_DIR / f"fig_gemnet_dftref_vs_dftrefboe_parity{suffix}.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE: {out}")

    metrics_out = TABLE_DIR / f"fig_gemnet_dftref_vs_dftrefboe_parity{suffix}_metrics.csv"
    pd.DataFrame(summary_rows).to_csv(metrics_out, index=False)
    print(f"WROTE: {metrics_out}")

    if missing:
        missing_out = INPUT_DIR / "missing_tmqm_prediction_inputs.txt"
        missing_out.write_text("\n".join(str(p) for p in missing) + "\n")
        print(f"WROTE: {missing_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qm9-prediction", choices=["mean_shift", "raw"], default="mean_shift")
    args = parser.parse_args()
    make_figure(qm9_prediction=args.qm9_prediction, allow_missing_tmqm=False)
