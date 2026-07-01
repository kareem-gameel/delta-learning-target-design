from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "manuscript" / "figures"
TABLE_DIR = ROOT / "manuscript" / "tables"


def read_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    variance = pd.read_csv(ROOT / "New_Analysis/datasets/dft_ref_boe_variance_and_boe_correlation.csv")
    all_row = variance.loc[variance["split"] == "all"].iloc[0]

    dft = pd.read_csv(
        ROOT / "New_Analysis/datasets/tmQM__DFT_ref-BoE.csv",
        usecols=["DFT_ref", "BoE_fit_DFT_ref", "DFT_ref-BoE"],
    )

    perf = pd.read_csv(ROOT / "New_Analysis/outputs/gemnet_dftref_vs_dftrefboe_tmqm_qm9_xtb_transfer_summary.csv")

    q_ref = pd.read_csv(
        ROOT
        / "New_Analysis/qm9_dft_ref_boe_transfer/outputs/requested_parity_against_DFT_ref/qm9_requested_parity_metrics_against_DFT_ref.csv"
    )
    q_boe = pd.read_csv(
        ROOT / "New_Analysis/qm9_dft_ref_boe_transfer/outputs/qm9_DFT_ref_BoE_gemnet_metrics.csv"
    )

    qm9_extra = {
        "DFT_ref_qm9_raw_mae": float(
            q_ref.loc[(q_ref["model_key"] == "gemnet_DFT") & (q_ref["calibration"] == "raw"), "mae"].iloc[0]
        ),
        "DFT_ref_qm9_shift_mae": float(
            q_ref.loc[
                (q_ref["model_key"] == "gemnet_DFT") & (q_ref["calibration"] == "mean_shift"), "mae"
            ].iloc[0]
        ),
        "DFT_ref_qm9_shift_r2": float(
            q_ref.loc[
                (q_ref["model_key"] == "gemnet_DFT") & (q_ref["calibration"] == "mean_shift"), "r2"
            ].iloc[0]
        ),
        "DFT_ref_qm9_shift_tau": float(
            q_ref.loc[
                (q_ref["model_key"] == "gemnet_DFT") & (q_ref["calibration"] == "mean_shift"), "kendall_tau"
            ].iloc[0]
        ),
        "DFT_ref_BoE_qm9_raw_mae": float(
            q_boe.loc[
                (q_boe["model_id"] == "gemnet_DFT_ref_BoE") & (q_boe["calibration"] == "raw"), "mae"
            ].iloc[0]
        ),
        "DFT_ref_BoE_qm9_shift_mae": float(
            q_boe.loc[
                (q_boe["model_id"] == "gemnet_DFT_ref_BoE")
                & (q_boe["calibration"] == "oracle_mean_shift"),
                "mae",
            ].iloc[0]
        ),
        "DFT_ref_BoE_qm9_shift_r2": float(
            q_boe.loc[
                (q_boe["model_id"] == "gemnet_DFT_ref_BoE")
                & (q_boe["calibration"] == "oracle_mean_shift"),
                "r2",
            ].iloc[0]
        ),
        "DFT_ref_BoE_qm9_shift_tau": float(
            q_boe.loc[
                (q_boe["model_id"] == "gemnet_DFT_ref_BoE")
                & (q_boe["calibration"] == "oracle_mean_shift"),
                "kendall_tau",
            ].iloc[0]
        ),
    }
    return all_row, dft, perf, qm9_extra


def get_perf(perf: pd.DataFrame, setting: str, target: str, calibration: str) -> float:
    row = perf.loc[
        (perf["setting"] == setting)
        & (perf["target_evaluated"] == target)
        & (perf["calibration"] == calibration)
    ].iloc[0]
    return float(row["mae"])


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def make_figure(all_row: pd.Series, dft: pd.DataFrame, perf: pd.DataFrame, qm9_extra: dict[str, float]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )

    blue = "#2F5D8C"
    orange = "#D95F02"

    fig = plt.figure(figsize=(9.3, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0])
    ax_hex = fig.add_subplot(gs[0, 0])
    ax_perf = fig.add_subplot(gs[0, 1])

    # Panel B: BoE fit correlation with reference-corrected DFT.
    hb = ax_hex.hexbin(
        dft["BoE_fit_DFT_ref"].to_numpy(),
        dft["DFT_ref"].to_numpy(),
        gridsize=65,
        bins="log",
        mincnt=1,
        cmap="viridis",
        linewidths=0.0,
    )
    lo = np.nanpercentile(np.r_[dft["BoE_fit_DFT_ref"].to_numpy(), dft["DFT_ref"].to_numpy()], 0.2)
    hi = np.nanpercentile(np.r_[dft["BoE_fit_DFT_ref"].to_numpy(), dft["DFT_ref"].to_numpy()], 99.8)
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad
    ax_hex.plot([lo, hi], [lo, hi], color="white", linewidth=1.2, alpha=0.95)
    ax_hex.plot([lo, hi], [lo, hi], color="#222222", linewidth=0.8, alpha=0.65, linestyle="--")
    ax_hex.set_xlim(lo, hi)
    ax_hex.set_ylim(lo, hi)
    ax_hex.set_xlabel(r"BoE prediction, $E_{\rm BoE}$ (Ha)")
    ax_hex.set_ylabel(r"Reference-corrected DFT, $E_{\rm ref}$ (Ha)")
    ax_hex.set_title("Composition baseline is highly predictive", pad=8)
    ax_hex.set_aspect("equal", adjustable="box")
    ax_hex.grid(False)
    for spine in ax_hex.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    cbar = fig.colorbar(hb, ax=ax_hex, fraction=0.045, pad=0.02)
    cbar.set_label("molecules", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax_hex.text(
        0.04,
        0.96,
        "\n".join(
            [
                fr"$R^2$ = {float(all_row['r2_DFTref_pred_by_BoEfit']):.3f}",
                fr"MAE = {float(all_row['mae_DFTref_pred_by_BoEfit']):.3f} Ha",
                fr"RMSE = {float(all_row['rmse_DFTref_pred_by_BoEfit']):.3f} Ha",
            ]
        ),
        transform=ax_hex.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    add_panel_label(ax_hex, "a")

    # Panel C: GemNet in-domain and OOD performance.
    tmqm_ref = get_perf(perf, "tmQM val in-domain", "DFT_ref", "native")
    tmqm_boe = get_perf(perf, "tmQM val in-domain", "DFT_ref,BoE", "native")
    qm9_ref = qm9_extra["DFT_ref_qm9_shift_mae"]
    qm9_boe = qm9_extra["DFT_ref_BoE_qm9_shift_mae"]
    x = np.arange(2)
    width = 0.34
    direct = [tmqm_ref, qm9_ref]
    boe = [tmqm_boe, qm9_boe]
    b1 = ax_perf.bar(x - width / 2, direct, width, color=blue, label=r"train $E_{\rm ref}$")
    b2 = ax_perf.bar(x + width / 2, boe, width, color=orange, label=r"train $E_{\rm ref,BoE}$")
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(["tmQM val\nin domain", "QM9 direct\nmean-shifted"])
    ax_perf.set_ylabel("GemNet MAE (Ha)")
    ax_perf.set_title("Target preprocessing improves GemNet", pad=8)
    style_axes(ax_perf)
    ax_perf.legend(frameon=False, loc="upper left")
    ax_perf.set_ylim(0, max(direct + boe) * 1.32)
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax_perf.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.001,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=0,
            )
    add_panel_label(ax_perf, "b")

    out = FIG_DIR / "fig_boe_target_preprocessing.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE: {out}")


def make_table(all_row: pd.Series, perf: pd.DataFrame, qm9_extra: dict[str, float]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "Training target": r"$E_{\rm ref}$",
            "Target variance\n(Ha²)": float(all_row["DFT_ref_var_pop"]),
            "Target std\n(Ha)": float(all_row["DFT_ref_std_pop"]),
            "tmQM val MAE\n(Ha)": get_perf(perf, "tmQM val in-domain", "DFT_ref", "native"),
            "QM9 raw MAE\n(Ha)": qm9_extra["DFT_ref_qm9_raw_mae"],
            "QM9 mean-shift\nMAE (Ha)": qm9_extra["DFT_ref_qm9_shift_mae"],
            "QM9 mean-shift\nR²": qm9_extra["DFT_ref_qm9_shift_r2"],
        },
        {
            "Training target": r"$E_{\rm ref,BoE}$",
            "Target variance\n(Ha²)": float(all_row["DFT_ref_BoE_var_pop"]),
            "Target std\n(Ha)": float(all_row["DFT_ref_BoE_std_pop"]),
            "tmQM val MAE\n(Ha)": get_perf(perf, "tmQM val in-domain", "DFT_ref,BoE", "native"),
            "QM9 raw MAE\n(Ha)": qm9_extra["DFT_ref_BoE_qm9_raw_mae"],
            "QM9 mean-shift\nMAE (Ha)": qm9_extra["DFT_ref_BoE_qm9_shift_mae"],
            "QM9 mean-shift\nR²": qm9_extra["DFT_ref_BoE_qm9_shift_r2"],
        },
    ]
    table = pd.DataFrame(rows)

    csv_table = table.copy()
    for col in csv_table.columns:
        if col != "Training target":
            csv_table[col] = csv_table[col].astype(float)
    csv_out = TABLE_DIR / "table_boe_target_preprocessing.csv"
    csv_table.to_csv(csv_out, index=False)
    print(f"WROTE: {csv_out}")

    tex_out = TABLE_DIR / "table_boe_target_preprocessing.tex"
    tex_table = table.copy()
    tex_table.to_latex(tex_out, index=False, escape=False, float_format="%.4g")
    print(f"WROTE: {tex_out}")

    display = table.copy()
    formats = {
        "Target variance\n(Ha²)": "{:.4g}",
        "Target std\n(Ha)": "{:.4f}",
        "tmQM val MAE\n(Ha)": "{:.4f}",
        "QM9 raw MAE\n(Ha)": "{:.4f}",
        "QM9 mean-shift\nMAE (Ha)": "{:.4f}",
        "QM9 mean-shift\nR²": "{:.3f}",
    }
    for col, fmt in formats.items():
        display[col] = display[col].map(lambda x: fmt.format(float(x)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(12.4, 2.55))
    ax.axis("off")
    ax.text(
        0,
        1.08,
        "Effect of Bag-of-Elements target preprocessing",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0,
        0.995,
        (
            "All target statistics use the aligned neutral tmQM set (n=71,042). "
            "QM9 direct-transfer metrics are shown before and after mean-shift calibration."
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#444444",
    )

    mpl_table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="lower center",
        bbox=[0, 0.0, 1, 0.78],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9.2)
    mpl_table.auto_set_column_width(col=list(range(len(display.columns))))

    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor("#d8d8d8")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#f0f2f4")
            cell.set_text_props(weight="bold", color="#1f1f1f")
            cell.set_height(0.19)
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#fafafa")
            cell.set_height(0.18)
            if col == 0:
                cell.set_text_props(weight="bold")

    ax.text(
        0,
        -0.09,
        (
            f"BoE fit to $E_{{\\rm ref}}$: Pearson r = {float(all_row['pearson_r_DFTref_vs_BoEfit']):.3f}, "
            f"$R^2$ = {float(all_row['r2_DFTref_pred_by_BoEfit']):.3f}, "
            f"MAE = {float(all_row['mae_DFTref_pred_by_BoEfit']):.4f} Ha."
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        color="#444444",
    )
    out = TABLE_DIR / "table_boe_target_preprocessing.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE: {out}")


def main() -> None:
    all_row, dft, perf, qm9_extra = read_inputs()
    make_figure(all_row, dft, perf, qm9_extra)
    make_table(all_row, perf, qm9_extra)


if __name__ == "__main__":
    main()
