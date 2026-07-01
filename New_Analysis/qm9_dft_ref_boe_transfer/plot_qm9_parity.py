#!/usr/bin/env python3
"""Make QM9 parity plots for DFT_ref,BoE transfer models.

The script expects outputs from run_qm9_dft_ref_boe_transfer.py:

  outputs/qm9_DFT_ref_BoE_gemnet_predictions.csv
  outputs/qm9_DFT_ref_BoE_gemnet_metrics.csv

It writes one PowerPoint-friendly 2x3 grid and individual panels for raw and
oracle mean-shift corrected predictions.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR / "outputs"
DEFAULT_PREDICTIONS = DEFAULT_OUT / "qm9_DFT_ref_BoE_gemnet_predictions.csv"
DEFAULT_METRICS = DEFAULT_OUT / "qm9_DFT_ref_BoE_gemnet_metrics.csv"

MODEL_ORDER = [
    "gemnet_DFT_ref_BoE",
    "gemnet_xTB_ref_BoE",
    "gemnet_Delta_xTB_ref_BoE",
]

MODEL_LABELS = {
    "gemnet_DFT_ref_BoE": "GemNet(DFT_ref,BoE)",
    "gemnet_xTB_ref_BoE": "GemNet(xTB_ref,BoE)",
    "gemnet_Delta_xTB_ref_BoE": "xTB_ref,BoE + GemNet(\u0394^xTB_ref,BoE)",
}

CALIBRATIONS = [
    ("raw", "Before mean shift", "y_pred_raw"),
    ("oracle_mean_shift", "After mean shift", "y_pred_mean_shift"),
]

PALETTE = {
    "gemnet_DFT_ref_BoE": (44, 107, 164),
    "gemnet_xTB_ref_BoE": (168, 84, 43),
    "gemnet_Delta_xTB_ref_BoE": (35, 132, 91),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-size", type=int, default=35_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=980)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(32, bold=True)
FONT_SUBTITLE = load_font(24, bold=True)
FONT_AXIS = load_font(22)
FONT_TICK = load_font(18)
FONT_LEGEND = load_font(19)
FONT_SMALL = load_font(16)


def require_file(path: Path, kind: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {kind}: {path}\n"
            "Download the Trillium outputs first, then rerun this script."
        )


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def nice_number(value: float) -> float:
    if value == 0 or not np.isfinite(value):
        return 1.0
    exp = math.floor(math.log10(abs(value)))
    frac = abs(value) / 10**exp
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.5:
        nice = 2.0
    elif frac < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return math.copysign(nice * 10**exp, value)


def ticks_for(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return [lo]
    step = nice_number(span / max(n - 1, 1))
    start = math.floor(lo / step) * step
    stop = math.ceil(hi / step) * step
    ticks = []
    val = start
    for _ in range(20):
        if val > stop + 0.5 * step:
            break
        if lo - 0.25 * step <= val <= hi + 0.25 * step:
            ticks.append(val)
        val += step
    return ticks


def fmt_tick(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pooled = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if pooled.size == 0:
        return -1.0, 1.0
    lo, hi = np.quantile(pooled, [0.001, 0.999])
    if hi <= lo:
        lo, hi = float(np.min(pooled)), float(np.max(pooled))
    pad = 0.07 * max(hi - lo, 1e-6)
    return float(lo - pad), float(hi + pad)


def project(values: np.ndarray, lo: float, hi: float, start: int, end: int, invert: bool = False) -> np.ndarray:
    scale = (end - start) / (hi - lo)
    pix = start + (values - lo) * scale
    if invert:
        pix = end - (values - lo) * scale
    return pix


def metric_row(metrics: pd.DataFrame, model_id: str, calibration: str) -> pd.Series:
    rows = metrics[(metrics["model_id"] == model_id) & (metrics["calibration"] == calibration)]
    if rows.empty:
        raise ValueError(f"Missing metrics for {model_id} / {calibration}")
    return rows.iloc[0]


def metric_text(row: pd.Series) -> list[str]:
    return [
        f"R2 = {row['r2']:.3f}",
        f"MAE = {row['mae']:.4f} Ha",
        f"RMSE = {row['rmse']:.4f} Ha",
        f"Tau = {row['kendall_tau']:.3f}",
    ]


def draw_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    model_id: str,
    calibration: str,
    title: str,
    pred_col: str,
    rng: np.random.Generator,
    sample_size: int,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    margin_left = 94 if show_ylabel else 58
    margin_right = 30
    margin_top = 92
    margin_bottom = 80 if show_xlabel else 48
    plot = (left + margin_left, top + margin_top, right - margin_right, bottom - margin_bottom)
    px0, py0, px1, py1 = plot

    subset = data[data["model_id"] == model_id]
    if subset.empty:
        raise ValueError(f"No prediction rows for {model_id}")

    x = subset["y_true"].to_numpy(float)
    y = subset[pred_col].to_numpy(float)
    lo, hi = axis_limits(x, y)

    draw.rectangle([left, top, right, bottom], fill=(255, 255, 255))
    draw.rectangle([px0, py0, px1, py1], fill=(249, 250, 251), outline=(215, 219, 224), width=1)

    ticks = ticks_for(lo, hi)
    for tick in ticks:
        tx = int(project(np.array([tick]), lo, hi, px0, px1)[0])
        ty = int(project(np.array([tick]), lo, hi, py0, py1, invert=True)[0])
        draw.line([(tx, py0), (tx, py1)], fill=(232, 235, 238), width=1)
        draw.line([(px0, ty), (px1, ty)], fill=(232, 235, 238), width=1)
        label = fmt_tick(tick)
        bbox = draw.textbbox((0, 0), label, font=FONT_TICK)
        draw.text((tx - (bbox[2] - bbox[0]) / 2, py1 + 10), label, fill=(64, 68, 72), font=FONT_TICK)
        if show_ylabel:
            draw.text((px0 - 12 - (bbox[2] - bbox[0]), ty - 10), label, fill=(64, 68, 72), font=FONT_TICK)

    draw.line([(px0, py1), (px1, py0)], fill=(41, 45, 50), width=2)

    valid = np.isfinite(x) & np.isfinite(y)
    indices = np.flatnonzero(valid)
    if indices.size > sample_size:
        indices = rng.choice(indices, size=sample_size, replace=False)
    sx = project(x[indices], lo, hi, px0, px1)
    sy = project(y[indices], lo, hi, py0, py1, invert=True)
    inside = (sx >= px0) & (sx <= px1) & (sy >= py0) & (sy <= py1)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    color = PALETTE[model_id]
    point = color + (58,)
    for xx, yy in zip(sx[inside].astype(int), sy[inside].astype(int)):
        odraw.ellipse([xx - 1, yy - 1, xx + 1, yy + 1], fill=point)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)

    label = MODEL_LABELS.get(model_id, model_id)
    draw.text((left + 24, top + 22), title, fill=(22, 26, 31), font=FONT_SUBTITLE)
    draw.text((left + 24, top + 54), label, fill=(73, 80, 87), font=FONT_SMALL)

    row = metric_row(metrics, model_id, calibration)
    lines = metric_text(row)
    maxw = max(draw.textbbox((0, 0), line, font=FONT_LEGEND)[2] for line in lines)
    legend_x = px1 - maxw - 24
    legend_y = py0 + 18
    legend_h = 28 * len(lines) + 20
    draw.rounded_rectangle(
        [legend_x - 14, legend_y - 12, px1 - 10, legend_y + legend_h],
        radius=8,
        fill=(255, 255, 255),
        outline=(218, 222, 226),
        width=1,
    )
    for i, line in enumerate(lines):
        draw.text((legend_x, legend_y + i * 28), line, fill=(25, 30, 35), font=FONT_LEGEND)

    if show_xlabel:
        xlabel = "QM9 DFT_ref,BoE target (Ha)"
        bbox = draw.textbbox((0, 0), xlabel, font=FONT_AXIS)
        draw.text((px0 + (px1 - px0 - (bbox[2] - bbox[0])) / 2, bottom - 40), xlabel, fill=(38, 43, 48), font=FONT_AXIS)
    if show_ylabel:
        ylabel = "Prediction (Ha)"
        tmp = Image.new("RGBA", (360, 44), (255, 255, 255, 0))
        tdraw = ImageDraw.Draw(tmp)
        tdraw.text((0, 4), ylabel, fill=(38, 43, 48), font=FONT_AXIS)
        tmp = tmp.rotate(90, expand=True)
        image.alpha_composite(tmp, (left + 18, py0 + (py1 - py0 - tmp.size[1]) // 2))


def make_grid(predictions: pd.DataFrame, metrics: pd.DataFrame, out_path: Path, sample_size: int, seed: int) -> None:
    width, height = 2550, 1700
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title = "QM9 transfer parity against DFT_ref,BoE"
    draw.text((70, 42), title, fill=(15, 20, 26), font=FONT_TITLE)
    draw.text(
        (70, 84),
        "Each panel reports metrics on all 130,807 molecules; points are a deterministic sample for readability.",
        fill=(88, 95, 104),
        font=FONT_AXIS,
    )

    x0, y0 = 50, 138
    gap_x, gap_y = 30, 36
    panel_w = (width - 2 * x0 - 2 * gap_x) // 3
    panel_h = (height - y0 - 54 - gap_y) // 2
    rng = np.random.default_rng(seed)
    for row_i, (calibration, cal_title, pred_col) in enumerate(CALIBRATIONS):
        for col_i, model_id in enumerate(MODEL_ORDER):
            left = x0 + col_i * (panel_w + gap_x)
            top = y0 + row_i * (panel_h + gap_y)
            draw_panel(
                image,
                (left, top, left + panel_w, top + panel_h),
                predictions,
                metrics,
                model_id,
                calibration,
                cal_title,
                pred_col,
                rng,
                sample_size,
                show_ylabel=col_i == 0,
                show_xlabel=row_i == 1,
            )

    image.convert("RGB").save(out_path, quality=96)


def make_individuals(predictions: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, width: int, height: int, sample_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for calibration, cal_title, pred_col in CALIBRATIONS:
        cal_dir = out_dir / ("parity_mean_shift" if calibration == "oracle_mean_shift" else "parity_raw")
        cal_dir.mkdir(parents=True, exist_ok=True)
        for model_id in MODEL_ORDER:
            image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            draw_panel(
                image,
                (0, 0, width, height),
                predictions,
                metrics,
                model_id,
                calibration,
                cal_title,
                pred_col,
                rng,
                sample_size,
                show_ylabel=True,
                show_xlabel=True,
            )
            image.convert("RGB").save(cal_dir / f"{slug(model_id)}.png", quality=96)


def main() -> None:
    args = parse_args()
    require_file(args.predictions, "predictions CSV")
    require_file(args.metrics, "metrics CSV")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    metrics = pd.read_csv(args.metrics)

    grid_path = args.out_dir / "qm9_DFT_ref_BoE_parity_grid_raw_and_mean_shift.png"
    make_grid(predictions, metrics, grid_path, args.sample_size, args.seed)
    make_individuals(predictions, metrics, args.out_dir, args.width, args.height, args.sample_size, args.seed)

    print("WROTE:", grid_path)
    print("WROTE:", args.out_dir / "parity_raw")
    print("WROTE:", args.out_dir / "parity_mean_shift")


if __name__ == "__main__":
    main()
