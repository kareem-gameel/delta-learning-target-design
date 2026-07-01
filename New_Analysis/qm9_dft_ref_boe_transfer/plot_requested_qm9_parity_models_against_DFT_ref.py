#!/usr/bin/env python3
"""PowerPoint-ready QM9 parity panels against DFT_ref.

This script uses legacy stacked QM9 GemNet outputs, which are already in the
DFT_ref frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs" / "requested_parity_against_DFT_ref"
STACKED_PRED = ROOT / "Target_Design" / "qm9_stacked_gemnet_transfer" / "outputs" / "qm9_stacked_predictions.csv"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    source_model_id: str
    source: str
    superscript: str
    suffix: str
    color: tuple[int, int, int]


MODELS = [
    ModelSpec(
        key="gemnet_DFT",
        source_model_id="gemnet_y",
        source="legacy_stacked_dft_ref",
        superscript="",
        suffix="",
        color=(44, 107, 164),
    ),
    ModelSpec(
        key="gemnet_Delta_xTB_plus_xTB",
        source_model_id="xtb_gemnet",
        source="legacy_stacked_dft_ref",
        superscript="xTB",
        suffix=" + xTB",
        color=(214, 106, 31),
    ),
    ModelSpec(
        key="gemnet_Delta_BoB_plus_BoB",
        source_model_id="bob_gemnet",
        source="legacy_stacked_dft_ref",
        superscript="BoB",
        suffix=" + BoB",
        color=(17, 17, 17),
    ),
    ModelSpec(
        key="gemnet_Delta_PRE_plus_PRE",
        source_model_id="b_short_gemnet",
        source="legacy_stacked_dft_ref",
        superscript="PRE",
        suffix=" + PRE",
        color=(47, 93, 155),
    ),
    ModelSpec(
        key="gemnet_Delta_PRE6p_conditioned_plus_PRE6p_conditioned",
        source_model_id="b_comp_long_cond_gemnet",
        source="legacy_stacked_dft_ref",
        superscript="PRE6p,cond",
        suffix=" + PRE6p,cond",
        color=(122, 92, 186),
    ),
]


def font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if serif:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_LABEL = font(58, bold=True)
FONT_LABEL_SMALL = font(34, bold=True)
FONT_LABEL_SERIF = font(68, bold=True, serif=True)
FONT_AXIS = font(54)
FONT_TICK = font(46)
FONT_LEGEND = font(46)
FONT_NOTE = font(38)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_legacy_model(model_id: str) -> pd.DataFrame:
    df = pd.read_csv(STACKED_PRED)
    sub = df[df["model_id"] == model_id].copy()
    if sub.empty:
        raise KeyError(f"Missing model_id={model_id} in {STACKED_PRED}")
    needed = ["mol_id", "reference_corrected_energy", "stacked_prediction_raw"]
    missing = [col for col in needed if col not in sub.columns]
    if missing:
        raise KeyError(f"{STACKED_PRED} missing columns: {missing}")
    out = sub[needed].dropna().copy()
    out["y_true"] = out["reference_corrected_energy"]
    out["y_pred_raw"] = out["stacked_prediction_raw"]
    return out[["mol_id", "y_true", "y_pred_raw"]].copy()


def combine_predictions() -> tuple[pd.DataFrame, list[str]]:
    require(STACKED_PRED)
    frames = []
    missing = []
    for spec in MODELS:
        try:
            df = load_legacy_model(spec.source_model_id)
        except Exception as exc:
            missing.append(f"{spec.key}: {exc}")
            continue
        df.insert(0, "model_key", spec.key)
        frames.append(df)
    if not frames:
        raise RuntimeError("No requested models could be loaded.")
    return pd.concat(frames, ignore_index=True), missing


def rank_values(values: np.ndarray) -> np.ndarray:
    uniques = {value: i + 1 for i, value in enumerate(sorted(set(values.tolist())))}
    return np.array([uniques[value] for value in values], dtype=np.int64)


def count_inversions(ranks: np.ndarray) -> int:
    n = int(ranks.max(initial=0))
    bit = np.zeros(n + 2, dtype=np.int64)

    def add(i: int) -> None:
        while i <= n:
            bit[i] += 1
            i += i & -i

    def prefix(i: int) -> int:
        total = 0
        while i > 0:
            total += int(bit[i])
            i -= i & -i
        return total

    inv = 0
    seen = 0
    for rank in ranks:
        inv += seen - prefix(int(rank))
        add(int(rank))
        seen += 1
    return int(inv)


def tie_pairs(values: np.ndarray) -> int:
    _, counts = np.unique(values, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def tie_pairs_xy(x: np.ndarray, y: np.ndarray) -> int:
    pairs = np.rec.fromarrays([x, y])
    _, counts = np.unique(pairs, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = x.size
    if n < 2:
        return float("nan")
    order = np.lexsort((y, x))
    sx = x[order]
    sy = y[order]
    ranks = rank_values(sy)
    discordant = count_inversions(ranks)
    n0 = n * (n - 1) // 2
    ties_x = tie_pairs(sx)
    ties_y = tie_pairs(sy)
    ties_xy = tie_pairs_xy(sx, sy)
    concordant = n0 - discordant - ties_x - ties_y + ties_xy
    denom = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    return float((concordant - discordant) / denom) if denom else float("nan")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    sse = float(np.sum(err * err))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": float(1.0 - sse / sst) if sst else float("nan"),
        "kendall_tau": kendall_tau_b(y_true, y_pred),
        "bias": float(np.mean(err)),
    }


def prepare_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    out_frames = []
    for key, sub in pred.groupby("model_key", sort=False):
        y_true = sub["y_true"].to_numpy(float)
        y_raw = sub["y_pred_raw"].to_numpy(float)
        shift = float(np.mean(y_true - y_raw))
        for calibration, y_pred, applied_shift in [
            ("raw", y_raw, 0.0),
            ("mean_shift", y_raw + shift, shift),
        ]:
            m = metrics(y_true, y_pred)
            rows.append({"model_key": key, "calibration": calibration, "shift": applied_shift, **m})
        tmp = sub.copy()
        tmp["y_pred_mean_shift"] = y_raw + shift
        out_frames.append(tmp)
    return pd.concat(out_frames, ignore_index=True), pd.DataFrame(rows)


def nice_tick_span(lo: float, hi: float) -> tuple[float, float, float]:
    span = max(hi - lo, 1e-9)
    raw_step = span / 5
    exp = math.floor(math.log10(raw_step))
    frac = raw_step / (10**exp)
    step = (1 if frac < 1.5 else 2 if frac < 3.5 else 5 if frac < 7.5 else 10) * 10**exp
    return math.floor(lo / step) * step, math.ceil(hi / step) * step, step


def fmt_tick(v: float) -> str:
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0]


def math_label_size(draw: ImageDraw.ImageDraw, spec: ModelSpec) -> int:
    if spec.key == "gemnet_DFT":
        return text_width(draw, "GemNet(DFT)", FONT_LABEL)
    return (
        text_width(draw, "GemNet(", FONT_LABEL)
        + text_width(draw, "Δ", FONT_LABEL_SERIF)
        + text_width(draw, spec.superscript, FONT_LABEL_SMALL)
        + text_width(draw, ")" + spec.suffix, FONT_LABEL)
    )


def draw_math_label(draw: ImageDraw.ImageDraw, x_center: int, y: int, spec: ModelSpec) -> None:
    if spec.key == "gemnet_DFT":
        label = "GemNet(DFT)"
        draw.text((x_center - text_width(draw, label, FONT_LABEL) / 2, y), label, font=FONT_LABEL, fill=(20, 25, 31))
        return
    x = x_center - math_label_size(draw, spec) / 2
    draw.text((x, y), "GemNet(", font=FONT_LABEL, fill=(20, 25, 31))
    x += text_width(draw, "GemNet(", FONT_LABEL)
    draw.text((x, y - 5), "Δ", font=FONT_LABEL_SERIF, fill=(20, 25, 31))
    x += text_width(draw, "Δ", FONT_LABEL_SERIF)
    draw.text((x, y - 9), spec.superscript, font=FONT_LABEL_SMALL, fill=(20, 25, 31))
    x += text_width(draw, spec.superscript, FONT_LABEL_SMALL)
    draw.text((x, y), ")" + spec.suffix, font=FONT_LABEL, fill=(20, 25, 31))


def draw_rotated_label(image: Image.Image, text: str, xy: tuple[int, int]) -> None:
    tmp = Image.new("RGBA", (620, 70), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    d.text((0, 12), text, font=FONT_AXIS, fill=(35, 42, 52))
    tmp = tmp.rotate(90, expand=True)
    image.alpha_composite(tmp, xy)


def plot_one(
    data: pd.DataFrame,
    metric_row: pd.Series,
    spec: ModelSpec,
    calibration: str,
    pred_col: str,
    out_path: Path,
    seed: int = 17,
    sample_size: int = 55_000,
) -> None:
    width, height = 1800, 1500
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    sub = data[data["model_key"] == spec.key].copy()
    y_true = sub["y_true"].to_numpy(float)
    y_pred = sub[pred_col].to_numpy(float)
    pooled = np.concatenate([y_true[np.isfinite(y_true)], y_pred[np.isfinite(y_pred)]])
    lo, hi = np.quantile(pooled, [0.001, 0.999])
    pad = 0.075 * max(hi - lo, 1e-6)
    lo, hi, step = nice_tick_span(float(lo - pad), float(hi + pad))

    left, right, top, bottom = 230, 105, 150, 230
    px0, py0, px1, py1 = left, top, width - right, height - bottom

    draw.rectangle([px0, py0, px1, py1], fill=(255, 255, 255), outline=(214, 219, 225), width=2)
    ticks = []
    val = lo
    while val <= hi + 0.5 * step:
        ticks.append(val)
        val += step
    for tick in ticks:
        x = px0 + (tick - lo) / (hi - lo) * (px1 - px0)
        y = py1 - (tick - lo) / (hi - lo) * (py1 - py0)
        draw.line([(x, py0), (x, py1)], fill=(236, 239, 243), width=1)
        draw.line([(px0, y), (px1, y)], fill=(236, 239, 243), width=1)
        label = fmt_tick(tick)
        bbox = draw.textbbox((0, 0), label, font=FONT_TICK)
        draw.text((x - (bbox[2] - bbox[0]) / 2, py1 + 18), label, font=FONT_TICK, fill=(78, 86, 96))
        draw.text((px0 - 20 - (bbox[2] - bbox[0]), y - 14), label, font=FONT_TICK, fill=(78, 86, 96))

    draw.line([(px0, py1), (px1, py0)], fill=(35, 42, 52), width=3)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    idx = np.flatnonzero(valid)
    if idx.size > sample_size:
        idx = np.random.default_rng(seed).choice(idx, size=sample_size, replace=False)
    xs = px0 + (y_true[idx] - lo) / (hi - lo) * (px1 - px0)
    ys = py1 - (y_pred[idx] - lo) / (hi - lo) * (py1 - py0)
    inside = (xs >= px0) & (xs <= px1) & (ys >= py0) & (ys <= py1)
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    point = spec.color + (105,)
    for x, y in zip(xs[inside].astype(int), ys[inside].astype(int)):
        odraw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=point)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)

    draw_math_label(draw, width // 2, 52, spec)

    xlabel = "True DFT (Ha)"
    bbox = draw.textbbox((0, 0), xlabel, font=FONT_AXIS)
    draw.text((px0 + (px1 - px0 - (bbox[2] - bbox[0])) / 2, height - 95), xlabel, font=FONT_AXIS, fill=(35, 42, 52))
    draw_rotated_label(image, "Prediction (Ha)", (44, py0 + (py1 - py0 - 620) // 2))
    draw = ImageDraw.Draw(image)

    lines = [
        f"R² = {metric_row['r2']:.3f}",
        f"MAE = {metric_row['mae']:.4f} Ha",
        f"RMSE = {metric_row['rmse']:.4f} Ha",
        f"τ = {metric_row['kendall_tau']:.3f}",
    ]
    maxw = max(text_width(draw, line, FONT_LEGEND) for line in lines)
    lx, ly = px0 + 56, py0 + 78
    draw.rounded_rectangle([lx - 30, ly - 28, lx + maxw + 36, ly + 214], radius=16, fill=(255, 255, 255), outline=(219, 224, 230), width=2)
    for i, line in enumerate(lines):
        draw.text((lx, ly + i * 58), line, font=FONT_LEGEND, fill=(20, 25, 31))
    if calibration == "mean_shift":
        note = "mean-shift corrected"
        draw.text((px0 + 28, py0 + 24), note, font=FONT_NOTE, fill=(106, 114, 124))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, quality=97)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions, missing = combine_predictions()
    predictions, metric_df = prepare_metrics(predictions)
    predictions.to_csv(OUT_DIR / "qm9_requested_parity_predictions_against_DFT_ref.csv", index=False)
    metric_df.to_csv(OUT_DIR / "qm9_requested_parity_metrics_against_DFT_ref.csv", index=False)
    (OUT_DIR / "missing_models.txt").write_text("\n".join(missing) + ("\n" if missing else ""))

    spec_by_key = {spec.key: spec for spec in MODELS}
    for calibration, pred_col in [("raw", "y_pred_raw"), ("mean_shift", "y_pred_mean_shift")]:
        cal_dir = OUT_DIR / calibration
        for _, row in metric_df[metric_df["calibration"] == calibration].iterrows():
            spec = spec_by_key[str(row["model_key"])]
            plot_one(predictions, row, spec, calibration, pred_col, cal_dir / f"{spec.key}.png")

    print("WROTE:", OUT_DIR)
    print(metric_df.to_string(index=False))
    if missing:
        print("\nMISSING:")
        for item in missing:
            print("-", item)


if __name__ == "__main__":
    main()
