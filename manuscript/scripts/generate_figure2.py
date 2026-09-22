#!/usr/bin/env python3
"""Regenerate only Figure 2; --refresh-data rebuilds its local source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "data" / "fig02_residual_design.csv"
BUNDLE = PACKAGE / "si/data/test_protocol_2026-09-22"
ROWS = [
    ("Delta_xTB-ref-BoE", "xTB", ""),
    ("Delta_DFTB-ref-BoE", "DFTB", ""),
    ("Delta_BoB-ref-BoE", "BoB", "linear"),
    ("Delta_MLP_BoB-ref-BoE", "BoB", "MLP"),
    ("Delta_PRE-ref-BoE", "PRE", "linear"),
    ("Delta_MLP_PRE-ref-BoE", "PRE", "MLP"),
    ("Delta_PRE_plus_A-ref-BoE", "PRE+A", "linear"),
    ("Delta_MLP_PRE_plus_A-ref-BoE", "PRE+A", "MLP"),
]
COLORS = {"xTB": "#00876C", "DFTB": "#B52B79", "BoB": "#3B75AF", "PRE": "#D68C0D", "PRE+A": "#C65137"}
DARK = "#253746"


def refresh_data() -> None:
    filenames = [
        "target_registry.csv", "roughness.csv", "baseline_metrics.csv",
        "tmqm_final_ledger.csv", "extended_metrics_per_seed.csv",
        "roughness_metadata.json", "tmqm_test_targets.csv",
    ]
    hashes = json.loads((BUNDLE / "bundle_sha256.json").read_text())
    for name in filenames:
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == hashes[name], name
    stats = pd.read_csv(BUNDLE / "roughness.csv")
    stats = stats.loc[stats.architecture.eq("GemNet")].set_index("target_id")
    baseline = pd.read_csv(BUNDLE / "baseline_metrics.csv")
    baseline = baseline.loc[baseline.split.eq("test")].set_index("target_id")
    seeds = pd.read_csv(BUNDLE / "tmqm_final_ledger.csv")
    seeds = seeds.loc[seeds.model.eq("gemnet")]
    assert seeds.test_status.eq("complete").all() and seeds.run_name.is_unique
    direct = seeds.loc[seeds.target_id.eq("E_ref-BoE"), "test_mae_hartree"]
    assert len(direct) == 3
    metrics = pd.read_csv(BUNDLE / "extended_metrics_per_seed.csv")
    paired_direct = metrics.query(
        'dataset=="tmqm" and cohort=="dftb_valid" and model=="gemnet" '
        'and target_id=="E_ref-BoE" and seed==101 and scale=="native"'
    )
    assert len(paired_direct) == 1
    targets = pd.read_csv(BUNDLE / "tmqm_test_targets.csv")
    test_targets = pd.read_csv(BUNDLE / "tmqm_test_targets.csv")
    primary = pd.read_csv(BUNDLE / "target_registry.csv").target_id.tolist()
    common = targets.dropna(subset=primary)
    assert len(common) == 7104 and targets.mol_id.is_unique and test_targets.mol_id.is_unique
    assert stats['split'].eq('test').all()
    records = []
    for target, family, fit in ROWS:
        supporting = family == "DFTB"
        match = seeds.loc[seeds.target_id.eq(target)]
        assert set(match.seed) == ({101} if supporting else {101, 102, 103})
        score = match.test_mae_hartree
        row = stats.loc[target]
        parent = stats.loc["E_ref-BoE_DFTB-valid" if supporting else "E_ref-BoE"]
        values = targets[target].dropna().to_numpy() if supporting else common[target].to_numpy()
        assert np.isfinite(values).all()
        np.testing.assert_allclose(np.subtract(*np.percentile(values, [75, 25])), row.target_iqr_hartree)
        np.testing.assert_allclose(test_targets[target].dropna().abs().mean(), baseline.loc[target, "mae"])
        assert int(baseline.loc[target, "n"]) == int(match.test_n.iloc[0])
        records.append({
            "target_id": target, "family": family, "fit": fit,
            "residual_iqr_Ha": row.target_iqr_hartree, "D_IQR": row.d_iqr,
            "standalone_baseline_mae_Ha": baseline.loc[target, "mae"],
            "gemnet_mae_mean_Ha": score.mean(), "gemnet_mae_sd_Ha": score.std(ddof=1),
            "n_seeds": len(score), "n_target_statistics": len(values),
            "n_test": int(match.test_n.iloc[0]), "parent_iqr_Ha": parent.target_iqr_hartree,
            "parent_D_IQR": parent.d_iqr,
            "direct_mae_mean_Ha": paired_direct.mae.iloc[0] if supporting else direct.mean(),
            "direct_mae_sd_Ha": np.nan if supporting else direct.std(ddof=1),
            "n_direct_seeds": 1 if supporting else len(direct),
            "evaluation_split": "test", "scale_and_roughness_split": "test",
            "supporting_control": supporting,
        })
    frame = pd.DataFrame(records)
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE, index=False)
    (SOURCE.parent / "fig02_provenance.json").write_text(json.dumps({
        "inputs": {name: {"path": str((BUNDLE / name).relative_to(PACKAGE)),
                          "sha256": hashes[name]} for name in filenames},
        "scale_and_roughness": "Corrected train-only targets; newly built 7104-molecule test graph, or 7068 DFTB-valid test molecules, normalized by the matching test IQR. No baseline or GNN refitting.",
        "standalone_baseline_mae": "Test mean absolute residual for each frozen baseline, on the same test IDs as its GNN.",
        "gnn_uncertainty": "Test MAE at each best-validation checkpoint; sample SD across seeds 101, 102, and 103. Canonical baseline fit seed remains 101.",
        "dftb_control": "Seed 101 only, with no seed SD; magenta dagger-labelled row. Local ticks use DFTB-valid parent IQR, roughness, and direct seed-101 test MAE. Training cohorts are not equalized.",
        "pooled_calculations": "None in this figure; DFTB is a display-only control, not included in primary-panel aggregates.",
    }, indent=2) + "\n")
    print(frame.to_string(index=False))


def draw() -> None:
    frame = pd.read_csv(SOURCE).fillna({"fit": ""})
    assert frame["target_id"].tolist() == [row[0] for row in ROWS]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8, "savefig.dpi": 450,
    })
    assert frame.evaluation_split.eq("test").all()
    y = np.array([7.9, 6.9, 5.65, 4.9, 3.65, 2.9, 1.65, 0.9])
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.03, 1.25]})
    parent = frame.iloc[0]
    references = [parent["parent_iqr_Ha"], parent["parent_D_IQR"], parent["direct_mae_mean_Ha"]]
    for index, ax in enumerate(axes):
        for low, high in [(4.43, 6.13), (0.43, 2.13)]:
            ax.axhspan(low, high, color="#F1F4F6", linewidth=0, zorder=0)
        ax.set_ylim(0.2, 8.55)
        ax.set_yticks(y)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#DCE3E8", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.axvline(references[index], color=DARK, linestyle="--", linewidth=1.15)
        ax.text(-0.09, 1.05, "abc"[index], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        label = r"$E_{\rm ref,BoE}$" if index < 2 else "Direct GemNet"
        ax.text(references[index], 8.34, label, color=DARK, fontsize=8.5,
                ha="center", va="bottom")
    axes[0].set_yticklabels([
        r"DFTB$^\dagger$" if row.family == "DFTB" else
        row.family if not row.fit else f"{row.family} ({row.fit})"
        for row in frame.itertuples()
    ])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    axes[0].get_yticklabels()[1].set_color(COLORS["DFTB"])

    for yy, row in zip(y, frame.itertuples(), strict=True):
        color = COLORS[row.family]
        axes[0].barh(yy, row.residual_iqr_Ha, height=0.46, color=color,
                     alpha=0.48 if row.fit == "linear" else 1, edgecolor=DARK, linewidth=0.65)
        inside = row.supporting_control
        axes[0].text(row.residual_iqr_Ha + (-0.002 if inside else 0.0018), yy,
                     f"{row.residual_iqr_Ha:.3f}", va="center", fontsize=8.5,
                     ha="right" if inside else "left", color="white" if inside else "black")
        axes[1].annotate("", xy=(row.D_IQR, yy), xytext=(row.parent_D_IQR, yy),
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1}, zorder=2)
        axes[1].scatter(row.D_IQR, yy, s=40, color=color, edgecolor=DARK, linewidth=0.65, zorder=3)
        axes[2].annotate("", xy=(row.gemnet_mae_mean_Ha, yy),
                         xytext=(row.standalone_baseline_mae_Ha, yy),
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2}, zorder=2)
        axes[2].scatter(row.standalone_baseline_mae_Ha, yy, s=38, color=color,
                        edgecolor=DARK, linewidth=0.65, zorder=3)
        axes[2].plot(row.gemnet_mae_mean_Ha, yy, linestyle="none", marker="D",
                     markersize=5.7, markerfacecolor="white", markeredgecolor=color,
                     markeredgewidth=1.25, zorder=4)
        if row.gemnet_mae_sd_Ha > 0:
            axes[2].errorbar(row.gemnet_mae_mean_Ha, yy, xerr=row.gemnet_mae_sd_Ha,
                             fmt="none", ecolor=color, elinewidth=1.1,
                             capsize=2, capthick=1.1, zorder=5)
        if row.supporting_control:
            for ax, reference in zip(axes, [row.parent_iqr_Ha, row.parent_D_IQR, row.direct_mae_mean_Ha]):
                ax.vlines(reference, yy - .22, yy + .22, colors=color, linewidth=1.2, zorder=6)
    axes[0].set_xlim(0, 0.097)
    axes[0].set_xticks([0, 0.03, 0.06, 0.09])
    axes[0].set_xlabel("Residual IQR (Ha)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.83, 26)
    axes[1].set_xticks([1, 2, 5, 10, 20], labels=["1", "2", "5", "10", "20"])
    axes[1].minorticks_off()
    axes[1].set_xlabel(r"Normalized roughness, $D_{\rm IQR}$")
    axes[2].set_xlim(0.003, 0.064)
    axes[2].set_xticks([0.01, 0.03, 0.05])
    axes[2].set_xlabel("Test MAE (Ha)")
    axes[2].axvspan(parent["direct_mae_mean_Ha"] - parent["direct_mae_sd_Ha"],
                    parent["direct_mae_mean_Ha"] + parent["direct_mae_sd_Ha"],
                    color=DARK, alpha=0.13, linewidth=0, zorder=1)
    axes[2].legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="#72818C", markeredgecolor=DARK,
               label="Baseline alone"),
        Line2D([], [], linestyle="none", marker="D", markerfacecolor="white", markeredgecolor="#72818C",
               label="GemNet on residual"),
    ], loc="lower right", bbox_to_anchor=(0.99, 0.04), frameon=False,
        fontsize=8.5, handletextpad=0.45, borderaxespad=0.25)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.89, bottom=0.15, wspace=0.24)
    fig.savefig(PACKAGE / "figures/fig02_residual_design.png", bbox_inches="tight")
    fig.savefig(PACKAGE / "figures/fig02_residual_design.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data()
    draw()
