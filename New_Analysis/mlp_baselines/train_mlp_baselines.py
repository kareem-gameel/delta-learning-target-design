#!/usr/bin/env python3
"""Train GPU MLP baselines on BoB, PRE, or PRE+A descriptors.

The first use case is fitting nonlinear baselines to DFT_ref,BoE:

    MLP_BoB_ref,BoE       = MLP(BoB -> DFT_ref,BoE)
    MLP_PRE_ref,BoE       = MLP(PRE -> DFT_ref,BoE)
    MLP_PRE+A_ref,BoE     = MLP(PRE+A -> DFT_ref,BoE)

and writing residual targets:

    Delta^MLP_BoB_ref,BoE   = DFT_ref,BoE - MLP_BoB_ref,BoE
    Delta^MLP_PRE_ref,BoE   = DFT_ref,BoE - MLP_PRE_ref,BoE
    Delta^MLP_PRE+A_ref,BoE = DFT_ref,BoE - MLP_PRE+A_ref,BoE

Preprocessing is intentionally split-aware:
- zero-variance feature filtering is fit on train only
- sparse feature scaling is fit on train only, without centering
- target normalization is fit on train only
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, diags
from scipy.stats import kendalltau
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
EXPANDED_PANEL_DIR = ROOT / "New_Analysis/expanded_target_panel"
if str(EXPANDED_PANEL_DIR) not in sys.path:
    sys.path.insert(0, str(EXPANDED_PANEL_DIR))

from expanded_panel_common import (  # noqa: E402
    DEFAULT_PRE_A_CACHE,
    DEFAULT_PRE_CACHE,
    DEFAULT_SPLIT_DIR,
    TARGET_BY_ID,
    TargetSpec,
    align_feature_block,
    apply_max_rows,
    load_feature_space,
    load_split_ids,
    load_target_frame,
)


MODEL_NAMES = {
    "bob": "MLP_BoB",
    "pre": "MLP_PRE",
    "pre_plus_a": "MLP_PRE_plus_A",
}

MODEL_DISPLAY_NAMES = {
    "bob": "MLP(BoB)",
    "pre": "MLP(PRE)",
    "pre_plus_a": "MLP(PRE+A)",
}

PRED_COLUMNS = {
    "bob": "MLP_BoB_ref-BoE",
    "pre": "MLP_PRE_ref-BoE",
    "pre_plus_a": "MLP_PRE_plus_A_ref-BoE",
}

DELTA_COLUMNS = {
    "bob": "Delta_MLP_BoB-ref-BoE",
    "pre": "Delta_MLP_PRE-ref-BoE",
    "pre_plus_a": "Delta_MLP_PRE_plus_A-ref-BoE",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_csv(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ["bob", "pre", "pre_plus_a"]
    items = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(items) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown feature spaces: {unknown}. Choices: {sorted(MODEL_NAMES)}")
    return items


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("--hidden-dims must contain positive integers, e.g. 512,128")
    return dims


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {split: df["split"].eq(split).to_numpy() for split in ["train", "val", "test"]}


def sparse_train_scale(x_train: csr_matrix, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x_train.mean(axis=0)).ravel()
    mean_sq = np.asarray(x_train.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean**2, 0.0)
    keep = var > eps
    scale = np.sqrt(var[keep])
    return keep, scale


def scale_with_train_stats(x: csr_matrix, keep: np.ndarray, scale: np.ndarray) -> csr_matrix:
    return (x[:, keep].tocsr() @ diags(1.0 / scale, offsets=0, format="csr")).tocsr()


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainResult:
    model_state: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    best_epoch: int
    best_val_mae: float
    train_target_mean: float
    train_target_std: float
    keep: np.ndarray
    scale: np.ndarray
    n_features_total: int
    n_features_retained: int
    n_features_dropped: int
    fit_minutes: float
    x_dense: torch.Tensor | None = None


def dense_cache_bytes(n_rows: int, n_cols: int) -> int:
    return int(n_rows) * int(n_cols) * np.dtype(np.float32).itemsize


def dense_cache_gb(n_rows: int, n_cols: int) -> float:
    return dense_cache_bytes(n_rows, n_cols) / (1024**3)


def build_dense_cache(
    x_scaled: csr_matrix,
    device: torch.device,
    mode: str,
    max_gb: float,
) -> torch.Tensor | None:
    if mode == "none":
        log("Dense cache disabled; batches will densify from CSR on demand.")
        return None
    estimated_gb = dense_cache_gb(x_scaled.shape[0], x_scaled.shape[1])
    if mode == "auto":
        if estimated_gb > max_gb:
            log(
                f"Dense cache skipped: estimated={estimated_gb:.2f} GiB exceeds "
                f"--max-dense-cache-gb={max_gb:.2f}."
            )
            return None
        target_device = device if device.type == "cuda" else torch.device("cpu")
    elif mode == "cuda":
        if device.type != "cuda":
            raise ValueError("--dense-cache cuda requires --device cuda or CUDA auto-detection.")
        target_device = device
    elif mode == "cpu":
        target_device = torch.device("cpu")
    else:
        raise ValueError("--dense-cache must be one of: auto, cuda, cpu, none")

    log(
        f"Building dense feature cache: rows={x_scaled.shape[0]} dims={x_scaled.shape[1]} "
        f"estimated={estimated_gb:.2f} GiB target={target_device}"
    )
    t0 = time.time()
    array = x_scaled.toarray().astype(np.float32, copy=False)
    tensor = torch.from_numpy(array)
    if target_device.type == "cuda":
        tensor = tensor.pin_memory().to(target_device, non_blocking=True)
    log(f"Built dense feature cache in {(time.time() - t0):.1f}s")
    return tensor


def dense_batch_from_csr(x: csr_matrix, rows: np.ndarray, device: torch.device, pin: bool) -> torch.Tensor:
    array = x[rows].toarray().astype(np.float32, copy=False)
    tensor = torch.from_numpy(array)
    if pin:
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=pin)


def feature_batch(
    x_scaled: csr_matrix,
    x_dense: torch.Tensor | None,
    rows: np.ndarray,
    device: torch.device,
    pin: bool,
) -> torch.Tensor:
    if x_dense is None:
        return dense_batch_from_csr(x_scaled, rows, device, pin)
    if x_dense.device.type == "cuda":
        idx = torch.as_tensor(rows, device=x_dense.device, dtype=torch.long)
        return x_dense.index_select(0, idx)
    tensor = x_dense[rows]
    if pin and not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=pin)


def predict_batches(
    model: nn.Module,
    x_scaled: csr_matrix,
    x_dense: torch.Tensor | None,
    rows: np.ndarray,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    pin = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            xb = feature_batch(x_scaled, x_dense, batch_rows, device, pin)
            pred_z = model(xb).detach().cpu().numpy()
            preds.append(pred_z * target_std + target_mean)
    return np.concatenate(preds) if preds else np.empty(0, dtype=float)


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def train_one(
    x: csr_matrix,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> TrainResult:
    t0 = time.time()
    train_rows = np.flatnonzero(masks["train"])
    val_rows = np.flatnonzero(masks["val"])
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError("Both train and val rows are required for MLP training.")

    keep, scale = sparse_train_scale(x[train_rows].tocsr(), args.zero_var_eps)
    if not keep.any():
        raise ValueError("No retained features after train-only variance filtering.")
    x_scaled = scale_with_train_stats(x, keep, scale)
    x_dense = build_dense_cache(x_scaled, device, args.dense_cache, args.max_dense_cache_gb)

    y_train = y[train_rows]
    target_mean = float(y_train.mean())
    target_std = float(y_train.std())
    if target_std <= 0:
        raise ValueError("Train target standard deviation is zero.")
    y_z = ((y - target_mean) / target_std).astype(np.float32)
    y_z_tensor = torch.from_numpy(y_z).to(device) if x_dense is not None and x_dense.device.type == "cuda" else None

    model = MLPRegressor(int(keep.sum()), args.hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(device, use_amp)
    pin = device.type == "cuda"
    rng = np.random.default_rng(args.seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        perm = rng.permutation(train_rows)
        epoch_losses = []
        for start in range(0, len(perm), args.batch_size):
            batch_rows = perm[start : start + args.batch_size]
            xb = feature_batch(x_scaled, x_dense, batch_rows, device, pin)
            if y_z_tensor is not None:
                batch_idx = torch.as_tensor(batch_rows, device=device, dtype=torch.long)
                yb = y_z_tensor.index_select(0, batch_idx)
            else:
                yb = torch.from_numpy(y_z[batch_rows])
                if pin:
                    yb = yb.pin_memory()
                yb = yb.to(device, non_blocking=pin)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))

        train_probe_rows = train_rows
        if args.train_eval_max_rows and len(train_rows) > args.train_eval_max_rows:
            train_probe_rows = np.sort(rng.choice(train_rows, size=args.train_eval_max_rows, replace=False))
        train_pred = predict_batches(
            model, x_scaled, x_dense, train_probe_rows, args.eval_batch_size, device, target_mean, target_std
        )
        val_pred = predict_batches(model, x_scaled, x_dense, val_rows, args.eval_batch_size, device, target_mean, target_std)
        train_mae = mae_np(y[train_probe_rows], train_pred)
        val_mae = mae_np(y[val_rows], val_pred)
        row = {
            "epoch": float(epoch),
            "train_loss_z": float(np.mean(epoch_losses)),
            "train_mae_probe": train_mae,
            "val_mae": val_mae,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        if val_mae + args.min_delta < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % args.log_every == 0 or stale_epochs == args.patience:
            log(
                "epoch={epoch} train_mae_probe={train:.6f} val_mae={val:.6f} "
                "best_val_mae={best:.6f} stale={stale}/{patience}".format(
                    epoch=epoch,
                    train=train_mae,
                    val=val_mae,
                    best=best_val_mae,
                    stale=stale_epochs,
                    patience=args.patience,
                )
            )

        if stale_epochs >= args.patience:
            log(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break

    return TrainResult(
        model_state=best_state,
        history=history,
        best_epoch=int(best_epoch),
        best_val_mae=float(best_val_mae),
        train_target_mean=target_mean,
        train_target_std=target_std,
        keep=keep,
        scale=scale,
        n_features_total=int(x.shape[1]),
        n_features_retained=int(keep.sum()),
        n_features_dropped=int((~keep).sum()),
        fit_minutes=round((time.time() - t0) / 60.0, 3),
        x_dense=x_dense,
    )


def metric_row(
    model_id: str,
    display_name: str,
    feature_space: str,
    target: TargetSpec,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_result: TrainResult,
    n_train: int,
    device: torch.device,
    hidden_dims: tuple[int, ...],
) -> dict[str, object]:
    resid = y_true - y_pred
    target_iqr = float(np.subtract(*np.percentile(y_true, [75, 25])))
    mae = mae_np(y_true, y_pred)
    tau = kendalltau(y_true, y_pred).statistic if len(y_true) > 1 else np.nan
    return {
        "model_id": model_id,
        "display_name": display_name,
        "feature_space": feature_space,
        "target_id": target.target_id,
        "target_label": target.target_label,
        "target_column": target.column,
        "split": split,
        "n": int(len(y_true)),
        "n_train": int(n_train),
        "mae": mae,
        "mae_over_iqr": float(mae / target_iqr) if target_iqr > 0 else np.nan,
        "rmse": rmse_np(y_true, y_pred),
        "r2": r2_np(y_true, y_pred),
        "kendall_tau": float(tau),
        "bias": float(np.mean(y_pred - y_true)),
        "residual_std": float(np.std(resid)),
        "target_std": float(np.std(y_true)),
        "target_iqr": target_iqr,
        "best_epoch": int(train_result.best_epoch),
        "best_val_mae": float(train_result.best_val_mae),
        "fit_minutes": float(train_result.fit_minutes),
        "n_features_total": int(train_result.n_features_total),
        "n_features_retained": int(train_result.n_features_retained),
        "n_features_dropped": int(train_result.n_features_dropped),
        "hidden_dims": ",".join(str(dim) for dim in hidden_dims),
        "device": str(device),
    }


def write_outputs(
    feature_space: str,
    target: TargetSpec,
    aligned: pd.DataFrame,
    y_pred: np.ndarray,
    masks: dict[str, np.ndarray],
    train_result: TrainResult,
    feature_meta: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    model_id = MODEL_NAMES[feature_space]
    display_name = MODEL_DISPLAY_NAMES[feature_space]
    pred_col = PRED_COLUMNS[feature_space]
    delta_col = DELTA_COLUMNS[feature_space]

    pred = aligned[["mol_id", "split", "target"]].copy()
    pred = pred.rename(columns={"target": target.column})
    pred["target_id"] = target.target_id
    pred["target_label"] = target.target_label
    pred["model_id"] = model_id
    pred[pred_col] = y_pred
    pred[delta_col] = pred[target.column] - pred[pred_col]
    pred["abs_error"] = pred[delta_col].abs()

    pred_path = args.out_dir / f"{model_id.lower()}_predictions_{target.target_id}.csv"
    pred.to_csv(pred_path, index=False)
    log(f"WROTE: {pred_path}")

    pred_resid_path = args.out_dir / f"tmQM__{model_id}_ref_BoE__pred_and_residual.csv"
    pred[["mol_id", target.column, pred_col, delta_col, "split"]].to_csv(pred_resid_path, index=False)
    log(f"WROTE: {pred_resid_path}")

    delta_path = args.out_dir / f"tmQM__Delta_{model_id}-ref-BoE.csv"
    pred[["mol_id", delta_col]].to_csv(delta_path, index=False)
    log(f"WROTE: {delta_path}")

    history_path = args.out_dir / f"{model_id.lower()}_training_history.csv"
    pd.DataFrame(train_result.history).to_csv(history_path, index=False)
    log(f"WROTE: {history_path}")

    preprocess_path = args.out_dir / f"{model_id.lower()}_preprocess.npz"
    np.savez_compressed(
        preprocess_path,
        keep=train_result.keep,
        scale=train_result.scale,
        train_target_mean=np.asarray([train_result.train_target_mean], dtype=float),
        train_target_std=np.asarray([train_result.train_target_std], dtype=float),
    )
    log(f"WROTE: {preprocess_path}")

    checkpoint_path = args.out_dir / f"{model_id.lower()}_model.pt"
    torch.save(
        {
            "model_id": model_id,
            "display_name": display_name,
            "target": asdict(target),
            "feature_space": feature_space,
            "feature_meta": feature_meta,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "state_dict": train_result.model_state,
            "train_target_mean": train_result.train_target_mean,
            "train_target_std": train_result.train_target_std,
            "best_epoch": train_result.best_epoch,
            "best_val_mae": train_result.best_val_mae,
            "args": vars(args),
        },
        checkpoint_path,
    )
    log(f"WROTE: {checkpoint_path}")

    rows = []
    y = aligned["target"].to_numpy(float)
    for split in ["train", "val", "test"]:
        if masks[split].any():
            rows.append(
                metric_row(
                    model_id=model_id,
                    display_name=display_name,
                    feature_space=str(feature_meta["feature_space"]),
                    target=target,
                    split=split,
                    y_true=y[masks[split]],
                    y_pred=y_pred[masks[split]],
                    train_result=train_result,
                    n_train=int(masks["train"].sum()),
                    device=device,
                    hidden_dims=args.hidden_dims,
                )
            )
    return rows


def run_feature_space(feature_space: str, target: TargetSpec, args: argparse.Namespace, device: torch.device):
    log(f"Loading feature space: {feature_space}")
    x_all, feature_ids, feature_meta = load_feature_space(feature_space, args.pre_cache, args.angular_cache)
    split_df = load_split_ids(args.split_dir)
    target_df = apply_max_rows(load_target_frame(target, split_df), args.max_rows_per_split)
    x, aligned = align_feature_block(x_all, feature_ids, target_df, "_feature_row")
    masks = split_masks(aligned)
    y = aligned["target"].to_numpy(float)

    validation = {
        "model_id": MODEL_NAMES[feature_space],
        "display_name": MODEL_DISPLAY_NAMES[feature_space],
        "feature_space": feature_space,
        "feature_meta": feature_meta,
        "target_id": target.target_id,
        "target_label": target.target_label,
        "target_csv": str(target.csv_path),
        "target_column": target.column,
        "rows_by_split": aligned["split"].value_counts().to_dict(),
        "feature_rows": int(x.shape[0]),
        "feature_dims": int(x.shape[1]),
        "device": str(device),
    }
    log(json.dumps(validation, indent=2))
    if args.validate_only:
        return validation, []

    log(f"TRAIN: {MODEL_DISPLAY_NAMES[feature_space]} -> {target.target_label}")
    train_result = train_one(x, y, masks, args, device)
    model = MLPRegressor(int(train_result.keep.sum()), args.hidden_dims, args.dropout).to(device)
    model.load_state_dict(train_result.model_state)
    x_scaled = scale_with_train_stats(x, train_result.keep, train_result.scale)
    y_pred = predict_batches(
        model=model,
        x_scaled=x_scaled,
        x_dense=train_result.x_dense,
        rows=np.arange(x_scaled.shape[0]),
        batch_size=args.eval_batch_size,
        device=device,
        target_mean=train_result.train_target_mean,
        target_std=train_result.train_target_std,
    )
    if not np.isfinite(y_pred).all():
        raise ValueError(f"Non-finite predictions for {MODEL_NAMES[feature_space]}")

    rows = write_outputs(feature_space, target, aligned, y_pred, masks, train_result, feature_meta, args, device)
    return validation, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-spaces", default="all", help="all or comma-list: bob,pre,pre_plus_a")
    parser.add_argument("--target-id", default="dft", choices=sorted(TARGET_BY_ID))
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--pre-cache", type=Path, default=DEFAULT_PRE_CACHE)
    parser.add_argument("--angular-cache", type=Path, default=DEFAULT_PRE_A_CACHE)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "New_Analysis/mlp_baselines/outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=parse_hidden_dims("512,128"))
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--huber-beta", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--dense-cache", default="auto", choices=["auto", "cuda", "cpu", "none"])
    parser.add_argument("--max-dense-cache-gb", type=float, default=16.0)
    parser.add_argument("--max-epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--zero-var-eps", type=float, default=1e-12)
    parser.add_argument("--train-eval-max-rows", type=int, default=10000)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    target = TARGET_BY_ID[args.target_id]
    feature_spaces = parse_csv(args.feature_spaces)

    log(f"device={device} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        log(f"cuda_name={torch.cuda.get_device_name(device)}")

    validations = []
    metric_rows = []
    for feature_space in feature_spaces:
        validation, rows = run_feature_space(feature_space, target, args, device)
        validations.append(validation)
        metric_rows.extend(rows)

    validation_path = args.out_dir / "mlp_input_validation.json"
    validation_path.write_text(json.dumps(validations, indent=2) + "\n")
    log(f"WROTE: {validation_path}")

    if args.validate_only:
        return

    metrics = pd.DataFrame(metric_rows)
    metrics_path = args.out_dir / "mlp_target_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    log(f"WROTE: {metrics_path}")


if __name__ == "__main__":
    main()
