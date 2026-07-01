import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path("/Users/kareem/PycharmProjects/deltaML_Paper")
BASE = ROOT / "trillium_project"
OUT = BASE / "qm9_composite_parity_outputs"
OUT.mkdir(exist_ok=True, parents=True)

LMDB_TOTAL = BASE / "lmdb_files/qm9_psi4_dft_ref-corr_neutral.lmdb"

RUNS = {
    "BoB+BoE_composite": {
        "npz": BASE / "qm9_predictions/qm9_pred_bob_boe_job302864.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_bob_on_boe_residuals.lmdb",
        "baseline_csv": ROOT / "4_Cross_Domain/bond_order/processed/QM9__BoB_tmQM_ridge_on_BoEresid__pred_and_residual.csv",
        "baseline_col": "E_BoE_plus_BoB_tmQM_on_QM9",
        "mode": "residual_plus_baseline",
    },
    "BoB_composite": {
        "npz": BASE / "qm9_predictions/qm9_pred_bob_job302890.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_bob_on_energy_residuals.lmdb",
        "baseline_csv": ROOT / "4_Cross_Domain/bond_order/processed/QM9__BoB_tmQM_ridge_on_energy__pred_and_residual.csv",
        "baseline_col": "E_BoB_tmQM_on_QM9",
        "mode": "residual_plus_baseline",
    },
    "BoE_composite": {
        "npz": BASE / "qm9_predictions/qm9_pred_boe_job302891.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_BOE_residuals.lmdb",
        "baseline_csv": ROOT / "4_Cross_Domain/fitting_BoE/processed/QM9__BoE_tmQM_ridge__residuals.csv",
        "baseline_col": "E_BoE_tmQM",
        "mode": "residual_plus_baseline",
    },
    "BoE_xTB_composite": {
        "npz": BASE / "qm9_predictions/qm9_pred_boe_xtb_job309204.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_boe_xtb_residuals.lmdb",
        "baseline_csv": ROOT / "4_Cross_Domain/fitting_BoE_xTB/processed/QM9__BoE_xTB_tmQM_ridge__residuals.csv",
        "baseline_col": "E_BoE_xTB_tmQM_on_QM9",
        "mode": "residual_plus_baseline",
    },
    "PRE_energy_composite": {
        "npz": BASE / "qm9_predictions/qm9_pred_pre_energy_job321090.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_pre_on_energy_residuals.lmdb",
        "baseline_csv": ROOT / "4_Cross_Domain/PRE_fitting/processed/QM9__PRE_tmQM_ridge_on_energy__residuals.csv",
        "baseline_col": "E_PRE_tmQM",
        "mode": "residual_plus_baseline",
    },
    "PRE_BoEresid_composite": {
        "npz": BASE / "qm9_predictions/qm9_pred_pre_boe_job321105.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_pre_on_boe_residuals.lmdb",
        "baseline_csv": ROOT / "4_Cross_Domain/PRE_fitting/processed/QM9__PRE_tmQM_ridge_on_BoEresid__residuals.csv",
        "baseline_col": "E_BoE_plus_PRE_tmQM",
        "mode": "residual_plus_baseline",
    },
    "xTB_composite": {
        "npz": BASE / "qm9_predictions/qm9_pred_xtb_job302893.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_psi4_deltaE.lmdb",
        "baseline_csv": ROOT / "Original_Data/ref-corrected_data/qm9_xtb_ref-corr.csv",
        "baseline_col": "reference_corrected_energy",
        "mode": "residual_plus_baseline",
    },
    "TotE_direct": {
        "npz": BASE / "qm9_predictions/qm9_pred_totE_job302892.npz",
        "resid_lmdb": BASE / "lmdb_files/qm9_psi4_dft_ref-corr_neutral.lmdb",
        "baseline_csv": None,
        "baseline_col": None,
        "mode": "direct_total",
    },
}


def load_lmdb_sid_maps(lmdb_path: Path):
    sid_to_molid = {}
    sid_to_y = {}
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k == b"length":
                continue
            d = pickle.loads(v)
            sid = int(d.sid)
            molid = str(getattr(d, "mol_id", sid))
            y = float(np.asarray(d.y_relaxed).reshape(-1)[0])
            sid_to_molid[sid] = molid
            sid_to_y[sid] = y
    env.close()
    return sid_to_molid, sid_to_y


def load_total_by_molid(total_lmdb_path: Path):
    molid_to_total = {}
    env = lmdb.open(str(total_lmdb_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k == b"length":
                continue
            d = pickle.loads(v)
            molid = str(getattr(d, "mol_id", int(d.sid)))
            molid_to_total[molid] = float(np.asarray(d.y_relaxed).reshape(-1)[0])
    env.close()
    return molid_to_total


def load_npz(npz_path: Path):
    z = np.load(npz_path)
    if "ids" not in z.files:
        raise ValueError(f"{npz_path} missing ids")
    if "energy" in z.files:
        pred = np.asarray(z["energy"]).reshape(-1).astype(np.float64)
    else:
        # fallback
        pred = None
        for k in z.files:
            arr = np.asarray(z[k])
            if np.issubdtype(arr.dtype, np.number) and arr.size > 1:
                pred = arr.reshape(-1).astype(np.float64)
                break
        if pred is None:
            raise ValueError(f"{npz_path} has no numeric prediction array")

    ids = np.array([int(x) for x in z["ids"].reshape(-1)], dtype=np.int64)
    n = min(len(ids), len(pred))
    return ids[:n], pred[:n]


def load_baseline_by_molid(csv_path: Path, baseline_col: str):
    df = pd.read_csv(csv_path)
    if "mol_id" not in df.columns:
        raise ValueError(f"{csv_path} missing mol_id")
    if baseline_col not in df.columns:
        raise ValueError(f"{csv_path} missing {baseline_col}")
    vals = pd.to_numeric(df[baseline_col], errors="coerce")
    out = {}
    for m, v in zip(df["mol_id"].astype(str), vals):
        if pd.notna(v):
            out[m] = float(v)
    return out


def compute_metrics(y_true, y_pred):
    residual = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    kt, _ = kendalltau(y_true, y_pred)
    rs = float(np.std(residual, ddof=1))
    return {
        "R2": float(r2),
        "MAE_Ha": float(mae),
        "RMSE_Ha": float(rmse),
        "Pearson_r": float(pr),
        "Spearman_rho": float(sr),
        "Kendall_tau": float(kt),
        "ResidualStd_Ha": rs,
    }


def make_plot(y_true, y_pred, m, title, out_png):
    TITLE_FS = 22
    LABEL_FS = 20
    TICK_FS = 16
    LEGEND_FS = 14

    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    ax.scatter(y_true, y_pred, s=10, alpha=0.25, edgecolors="none")

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.6)

    ax.set_xlabel("True Total Energy (Ha)", fontsize=LABEL_FS)
    ax.set_ylabel("Predicted Total Energy (Ha)", fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)

    handles = [
        Line2D([], [], lw=0, label=f"R² = {m['R2']:.4f}"),
        Line2D([], [], lw=0, label=f"MAE = {m['MAE_Ha']:.6f} Ha"),
        Line2D([], [], lw=0, label=f"RMSE = {m['RMSE_Ha']:.6f} Ha"),
        Line2D([], [], lw=0, label=f"Kendall τ = {m['Kendall_tau']:.4f}"),
        Line2D([], [], lw=0, label=f"Residual std = {m['ResidualStd_Ha']:.6f} Ha"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=LEGEND_FS, handlelength=0, handletextpad=0)
    leg.get_texts()[3].set_color("tab:blue")
    leg.get_texts()[4].set_color("tab:blue")

    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=260)
    plt.close(fig)


total_by_molid = load_total_by_molid(LMDB_TOTAL)

all_rows = []
rank_rows = []
debug_rows = []
skipped_rows = []

for model, cfg in RUNS.items():
    required = [cfg["npz"], cfg["resid_lmdb"]]
    if cfg["mode"] == "residual_plus_baseline":
        required.append(cfg["baseline_csv"])
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        skipped_rows.append(
            {"Model": model, "reason": "missing_inputs", "missing_paths": " | ".join(missing)}
        )
        print(f"Skipping {model}: missing inputs")
        for mp in missing:
            print("  -", mp)
        continue

    ids, pred = load_npz(cfg["npz"])
    sid_to_molid_resid, _ = load_lmdb_sid_maps(cfg["resid_lmdb"])

    base_by_molid = None
    if cfg["mode"] == "residual_plus_baseline":
        base_by_molid = load_baseline_by_molid(cfg["baseline_csv"], cfg["baseline_col"])

    y_true = []
    y_pred = []
    miss_sid = 0
    miss_total = 0
    miss_base = 0

    for sid, p in zip(ids, pred):
        molid = sid_to_molid_resid.get(int(sid))
        if molid is None:
            miss_sid += 1
            continue

        t = total_by_molid.get(molid)
        if t is None:
            miss_total += 1
            continue

        if cfg["mode"] == "residual_plus_baseline":
            b = base_by_molid.get(molid)
            if b is None:
                miss_base += 1
                continue
            yp = float(b) + float(p)
        else:
            yp = float(p)

        y_true.append(float(t))
        y_pred.append(yp)

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        raise RuntimeError(f"{model}: no aligned points")

    m = compute_metrics(y_true, y_pred)
    all_rows.append({
        "Model": model,
        "N_aligned": len(y_true),
        "N_npz": len(ids),
        "missing_sid": miss_sid,
        "missing_total": miss_total,
        "missing_baseline": miss_base,
        **m,
    })
    rank_rows.append({
        "Model": model,
        "Pearson_r": m["Pearson_r"],
        "Spearman_rho": m["Spearman_rho"],
        "Kendall_tau": m["Kendall_tau"],
    })
    debug_rows.append({
        "Model": model,
        "N_aligned": len(y_true),
        "N_npz": len(ids),
        "missing_sid": miss_sid,
        "missing_total": miss_total,
        "missing_baseline": miss_base,
    })

    out_png = OUT / f"parity_qm9_composite_{model.replace('+', 'plus')}.png"
    make_plot(y_true, y_pred, m, f"QM9 Composite Parity: {model}", out_png)

pd.DataFrame(all_rows).sort_values("Model").to_csv(OUT / "qm9_composite_metrics_all.csv", index=False)
pd.DataFrame(rank_rows).sort_values("Model").to_csv(OUT / "qm9_composite_ranking_scores.csv", index=False)
pd.DataFrame(debug_rows).sort_values("Model").to_csv(OUT / "qm9_composite_alignment_debug.csv", index=False)
if skipped_rows:
    pd.DataFrame(skipped_rows).sort_values("Model").to_csv(OUT / "qm9_composite_skipped_runs.csv", index=False)

print("Saved:")
print(OUT / "qm9_composite_metrics_all.csv")
print(OUT / "qm9_composite_ranking_scores.csv")
print(OUT / "qm9_composite_alignment_debug.csv")
if skipped_rows:
    print(OUT / "qm9_composite_skipped_runs.csv")
for p in sorted(OUT.glob("parity_qm9_composite_*.png")):
    print(p)
