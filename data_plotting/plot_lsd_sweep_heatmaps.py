#!/usr/bin/env python3
"""Heatmaps for LSD beta / w_min sweep CSVs."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

from overview_plotting import (
    Record,
    configure_matplotlib,
    ensure_output_dir,
    kv_label,
    make_hashable,
    read_records,
    safe_filename,
    sort_key,
    value_label,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_KEYS = {"lsd_beta", "lsd_w_min"}
DROP_KEYS = {"sigmas"}
SIGMA_SCALE = math.sqrt(2.0 * math.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2D heatmaps for LSD parameter sweeps. Each page fixes sigma "
            "and all metadata except lsd_beta and lsd_w_min."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "results/bivariate_bicycle/bivariate_bicycle_lsd_sweep.csv",
        help="Input sweep CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/figures/lsd_sweep_heatmaps",
        help="Directory for the PDF and index CSV.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input CSV stem.",
    )
    parser.add_argument(
        "--linear-color",
        action="store_true",
        help="Use linear color scale instead of log color scale.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each populated cell with the error rate.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of heatmap pages, useful for quick previews.",
    )
    return parser.parse_args()


def config_key(record: Record) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted(
            (
                key,
                make_hashable(value),
            )
            for key, value in record.metadata.items()
            if key not in SWEEP_KEYS and key not in DROP_KEYS
        )
    )


def config_dict(key: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(key)


def config_title(config: dict[str, Any]) -> str:
    preferred = [
        "code_name",
        "sigma",
        "basis",
        "local_search",
        "sphere_decoding",
        "balance_weights",
        "schedule",
    ]
    pieces = []
    for key in preferred:
        if key not in config:
            continue
        value = config[key]
        if key == "sigma":
            pieces.append(r"$\sqrt{2\pi}\sigma$=" + f"{SIGMA_SCALE * float(value):.4g}")
        else:
            pieces.append(kv_label(key, value))
    return ", ".join(pieces) if pieces else "LSD sweep configuration"


def config_filename(config_id: int, config: dict[str, Any]) -> str:
    parts = [f"config_{config_id:04d}"]
    for key in ("code_name", "basis", "sigma", "local_search", "sphere_decoding", "balance_weights"):
        if key in config:
            value = config[key]
            if key == "sigma":
                value = f"{SIGMA_SCALE * float(value):.4g}"
            parts.append(f"{key}_{value_label(value)}")
    return safe_filename("_".join(parts)) + ".png"


def cell_edges(values: list[float]) -> np.ndarray:
    """Infer pcolormesh cell edges from sorted, possibly nonuniform centers."""
    centres = np.asarray(values, dtype=float)
    if centres.size == 0:
        raise ValueError("cannot infer edges for an empty axis")
    if centres.size == 1:
        width = max(abs(float(centres[0])) * 0.1, 0.5)
        return np.array([centres[0] - width, centres[0] + width], dtype=float)

    mids = (centres[:-1] + centres[1:]) / 2.0
    first = centres[0] - (mids[0] - centres[0])
    last = centres[-1] + (centres[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def tick_values(values: list[float], max_ticks: int = 9) -> list[float]:
    if len(values) <= max_ticks:
        return values
    indices = np.linspace(0, len(values) - 1, max_ticks, dtype=int)
    return [values[int(index)] for index in np.unique(indices)]


def plot_config_heatmap(
    records: list[Record],
    config: dict[str, Any],
    *,
    linear_color: bool,
    annotate: bool,
) -> tuple[plt.Figure, dict[str, Any]]:
    betas = sorted({float(record.metadata["lsd_beta"]) for record in records})
    w_mins = sorted({float(record.metadata["lsd_w_min"]) for record in records})
    beta_index = {value: idx for idx, value in enumerate(betas)}
    w_min_index = {value: idx for idx, value in enumerate(w_mins)}

    observed_rates = np.full((len(w_mins), len(betas)), np.nan)
    plot_rates = np.full((len(w_mins), len(betas)), np.nan)
    shots = np.zeros_like(observed_rates)
    errors = np.zeros_like(observed_rates)

    for record in records:
        x = beta_index[float(record.metadata["lsd_beta"])]
        y = w_min_index[float(record.metadata["lsd_w_min"])]
        observed_rates[y, x] = record.error_rate
        plot_rates[y, x] = record.plot_rate
        shots[y, x] = record.shots
        errors[y, x] = record.errors

    populated = plot_rates[np.isfinite(plot_rates)]
    if populated.size == 0:
        raise ValueError("configuration has no populated sweep cells")

    fig, ax = plt.subplots(figsize=(7.5, 5.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f3f3f3")

    if linear_color:
        image = ax.pcolormesh(
            cell_edges(betas),
            cell_edges(w_mins),
            observed_rates,
            cmap=cmap,
            shading="flat",
        )
        cbar_label = "Logical error rate"
    else:
        positive = populated[populated > 0]
        vmin = max(float(np.min(positive)), 1e-12) if positive.size else 1e-12
        vmax = max(float(np.max(positive)), vmin * 1.01) if positive.size else 1.0
        image = ax.pcolormesh(
            cell_edges(betas),
            cell_edges(w_mins),
            plot_rates,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="flat",
        )
        cbar_label = "Logical error rate (log scale; zero cells use 0.5/shots)"

    x_ticks = tick_values(betas)
    y_ticks = tick_values(w_mins)
    ax.set_xticks(x_ticks, [f"{value:g}" for value in x_ticks])
    ax.set_yticks(y_ticks, [f"{value:g}" for value in y_ticks])
    ax.set_xlabel(r"LSD $\beta$")
    ax.set_ylabel(r"LSD $w_{\min}$")
    ax.set_title(config_title(config), fontsize=10)
    ax.set_xlim(cell_edges(betas)[0], cell_edges(betas)[-1])
    ax.set_ylim(cell_edges(w_mins)[0], cell_edges(w_mins)[-1])
    ax.grid(False)

    if annotate:
        for y in range(observed_rates.shape[0]):
            for x in range(observed_rates.shape[1]):
                if np.isfinite(observed_rates[y, x]):
                    ax.text(betas[x], w_mins[y], f"{observed_rates[y, x]:.2g}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)

    best_y, best_x = np.unravel_index(np.nanargmin(observed_rates), observed_rates.shape)
    summary = {
        "records": len(records),
        "cells": int(np.isfinite(observed_rates).sum()),
        "beta_min": min(betas),
        "beta_max": max(betas),
        "w_min_min": min(w_mins),
        "w_min_max": max(w_mins),
        "best_beta": betas[best_x],
        "best_w_min": w_mins[best_y],
        "best_error_rate": float(observed_rates[best_y, best_x]),
        "best_plot_error_rate": float(plot_rates[best_y, best_x]),
        "best_errors": int(errors[best_y, best_x]),
        "best_shots": int(shots[best_y, best_x]),
    }
    return fig, summary


def write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    metadata_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key
            not in {
                "config_id",
                "records",
                "cells",
                "beta_min",
                "beta_max",
                "w_min_min",
                "w_min_max",
                "best_beta",
                "best_w_min",
                "best_error_rate",
                "best_plot_error_rate",
                "best_errors",
                "best_shots",
                "png",
            }
        }
    )
    fields = [
        "config_id",
        "records",
        "cells",
        "beta_min",
        "beta_max",
        "w_min_min",
        "w_min_max",
        "best_beta",
        "best_w_min",
        "best_error_rate",
        "best_plot_error_rate",
        "best_errors",
        "best_shots",
        "png",
        *metadata_keys,
    ]

    import csv

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    records = [
        record
        for record in read_records(args.input)
        if "lsd_beta" in record.metadata and "lsd_w_min" in record.metadata
    ]
    if not records:
        raise SystemExit(f"No LSD sweep records found in {args.input}")

    groups: dict[tuple[tuple[str, Any], ...], list[Record]] = defaultdict(list)
    for record in records:
        groups[config_key(record)].append(record)

    output_dir = ensure_output_dir(args.output_dir)
    prefix = args.prefix or safe_filename(args.input.stem)
    pdf_path = output_dir / f"{prefix}_heatmaps.pdf"
    index_path = output_dir / f"{prefix}_heatmap_index.csv"
    png_dir = ensure_output_dir(output_dir / f"{prefix}_heatmaps")

    rows: list[dict[str, Any]] = []
    group_items = sorted(groups.items(), key=lambda item: tuple((key, sort_key(value)) for key, value in item[0]))
    if args.max_pages is not None:
        group_items = group_items[: args.max_pages]

    with PdfPages(pdf_path) as pdf:
        for config_id, (key, config_records) in enumerate(group_items, start=1):
            config = config_dict(key)
            fig, summary = plot_config_heatmap(
                config_records,
                config,
                linear_color=args.linear_color,
                annotate=args.annotate,
            )
            png_name = config_filename(config_id, config)
            fig.savefig(png_dir / png_name)
            pdf.savefig(fig)
            plt.close(fig)

            rows.append(
                {
                    "config_id": config_id,
                    "png": str(png_dir / png_name),
                    **summary,
                    **{key: value_label(value) for key, value in config.items()},
                }
            )

    write_index(index_path, rows)
    print(f"Wrote {len(rows)} heatmaps to {pdf_path}")
    print(f"Wrote index to {index_path}")


if __name__ == "__main__":
    main()
