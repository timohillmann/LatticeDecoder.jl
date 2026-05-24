#!/usr/bin/env python3
"""Overview plotting helpers for Sinter-style decoder CSV files.

These plots are intentionally configuration catalogues.  They are meant to make
it easy to see which simulated settings are promising before hand-selecting
curves for paper figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotting_lib as pl
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGMA_SCALE = math.sqrt(2.0 * math.pi)

PALETTE = list(pl.colors_rsb)
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "h"]


@dataclass(frozen=True)
class Record:
    shots: int
    errors: int
    seconds: float
    decoder: str
    metadata: dict[str, Any]

    @property
    def sigma_scaled(self) -> float:
        return SIGMA_SCALE * float(self.metadata["sigma"])

    @property
    def error_rate(self) -> float:
        return self.errors / self.shots if self.shots else float("nan")

    @property
    def plot_rate(self) -> float:
        if not self.shots:
            return float("nan")
        return self.error_rate if self.errors else 0.5 / self.shots


def configure_matplotlib() -> None:
    pl.update_settings(True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.45,
        }
    )


def read_records(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            records.append(
                Record(
                    shots=int(row["shots"]),
                    errors=int(row["errors"]),
                    seconds=float(row["seconds"]),
                    decoder=row["decoder"],
                    metadata=json.loads(row["json_metadata"]),
                )
            )
    return aggregate_records(records)


def aggregate_records(records: Iterable[Record]) -> list[Record]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        key = json.dumps(record.metadata, sort_keys=True, separators=(",", ":"))
        if key not in buckets:
            buckets[key] = {
                "shots": 0,
                "errors": 0,
                "seconds": 0.0,
                "decoder": record.decoder,
                "metadata": record.metadata,
            }
        bucket = buckets[key]
        bucket["shots"] += record.shots
        bucket["errors"] += record.errors
        bucket["seconds"] += record.seconds

    return [
        Record(
            shots=bucket["shots"],
            errors=bucket["errors"],
            seconds=bucket["seconds"],
            decoder=bucket["decoder"],
            metadata=bucket["metadata"],
        )
        for bucket in buckets.values()
    ]


def metadata_value(record: Record, key: str, default: Any = None) -> Any:
    return record.metadata.get(key, default)


def value_label(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return "[" + ",".join(value_label(v) for v in value) + "]"
    return str(value)


def kv_label(key: str, value: Any) -> str:
    clean = key.replace("_", " ")
    return f"{clean}={value_label(value)}"


def config_tuple(record: Record, keys: list[str]) -> tuple[Any, ...]:
    return tuple(make_hashable(metadata_value(record, key)) for key in keys)


def make_hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    return value


def sort_key(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        return ("bool", int(value))
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(sort_key(v) for v in value))
    return ("str", str(value))


def default_record_sort(record: Record, group_key: str) -> tuple[Any, ...]:
    return (
        sort_key(metadata_value(record, group_key)),
        record.sigma_scaled,
        record.errors,
        record.shots,
    )


def wilson_interval(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    if shots <= 0:
        return (float("nan"), float("nan"))
    phat = errors / shots
    denom = 1.0 + z * z / shots
    centre = (phat + z * z / (2.0 * shots)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * shots)) / shots) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config_index(
    path: Path,
    config_rows: list[dict[str, Any]],
    config_keys: list[str],
    x_index_label: str = "sigma",
) -> None:
    fieldnames = [
        "config_id",
        "records",
        "curves",
        f"{x_index_label}_min",
        f"{x_index_label}_max",
        "min_error_rate",
        "max_error_rate",
        *config_keys,
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in config_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_family_catalogue(
    *,
    input_csv: Path,
    output_dir: Path,
    family_name: str,
    group_key: str,
    group_label: Callable[[Any], str] | None = None,
    config_keys: list[str],
    title_prefix: str,
    x_label: str = r"Noise strength $\sqrt{2\pi}\sigma$",
    y_label: str = "Logical error rate",
    x_value_func: Callable[[Record], float] | None = None,
    x_index_label: str = "sigma",
    max_legend_items: int = 14,
    xlim_l = None,
    xlim_r = None,
) -> None:
    configure_matplotlib()
    records = read_records(input_csv)
    x_value = x_value_func or (lambda record: record.sigma_scaled)
    output_dir = ensure_output_dir(output_dir)

    configs: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    for record in records:
        configs[config_tuple(record, config_keys)].append(record)

    config_items = sorted(configs.items(), key=lambda item: tuple(sort_key(v) for v in item[0]))
    index_rows: list[dict[str, Any]] = []
    combined_pdf = output_dir / f"{safe_filename(family_name)}_configuration_catalogue.pdf"

    with PdfPages(combined_pdf) as pdf:
        for config_id, (config, config_records) in enumerate(config_items, start=1):
            fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
            groups: dict[Any, list[Record]] = defaultdict(list)
            for record in config_records:
                groups[make_hashable(metadata_value(record, group_key))].append(record)

            for idx, (group, group_records) in enumerate(
                sorted(groups.items(), key=lambda item: sort_key(item[0]))
            ):
                group_records = sorted(group_records, key=lambda r: x_value(r))
                xs = np.array([x_value(r) for r in group_records])
                ys = np.array([r.plot_rate for r in group_records])
                lowers = []
                uppers = []
                for r, y in zip(group_records, ys):
                    lo, hi = wilson_interval(r.errors, r.shots)
                    lo = max(lo, 0.5 / r.shots) if r.errors == 0 else lo
                    lowers.append(max(0.0, y - lo))
                    uppers.append(max(0.0, hi - y))

                label = group_label(group) if group_label else f"{group_key}={value_label(group)}"
                ax.errorbar(
                    xs,
                    ys,
                    yerr=np.array([lowers, uppers]),
                    color=PALETTE[idx % len(PALETTE)],
                    marker=MARKERS[idx % len(MARKERS)],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    capsize=2.0,
                    label=label,
                )

            subtitle = ", ".join(kv_label(key, value) for key, value in zip(config_keys, config))
            ax.set_title(f"{title_prefix} - config {config_id:03d}\n{subtitle}", loc="left")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_yscale("log")
            ax.set_ylim(bottom=min(3e-5, min(r.plot_rate for r in config_records) * 0.6), top=1.05)
            ax.set_xlim(
                left=xlim_l if xlim_l is not None else min(x_value(r) for r in config_records) - 0.02,
                right=xlim_r if xlim_r is not None else max(x_value(r) for r in config_records) + 0.02,
            )
            ax.tick_params(direction="out", length=3)
            if len(groups) <= max_legend_items:
                ax.legend(title=group_key.replace("_", " "), frameon=False, ncols=2)
            else:
                ax.text(
                    0.01,
                    0.02,
                    f"{len(groups)} curves; see config index",
                    transform=ax.transAxes,
                    fontsize=8,
                    alpha=0.75,
                )

            png_path = output_dir / f"config_{config_id:03d}_{safe_filename(subtitle)[:120]}.png"
            pdf_path = output_dir / f"config_{config_id:03d}_{safe_filename(subtitle)[:120]}.pdf"
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            rates = [r.error_rate for r in config_records]
            index_row = {
                "config_id": f"{config_id:03d}",
                "records": len(config_records),
                "curves": len(groups),
                f"{x_index_label}_min": f"{min(x_value(r) for r in config_records):.6g}",
                f"{x_index_label}_max": f"{max(x_value(r) for r in config_records):.6g}",
                "min_error_rate": f"{min(rates):.6g}",
                "max_error_rate": f"{max(rates):.6g}",
            }
            index_row.update({key: value_label(value) for key, value in zip(config_keys, config)})
            index_rows.append(index_row)

    write_config_index(output_dir / "configuration_index.csv", index_rows, config_keys, x_index_label)
    print(f"Wrote {len(config_items)} configuration plots to {output_dir}")
    print(f"Wrote combined catalogue: {combined_pdf}")


def parse_common_args(
    *,
    default_input: str,
    default_output: str,
    description: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / default_input,
        help="Sinter-style CSV file to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / default_output,
        help="Directory for PNG/PDF figures and configuration_index.csv.",
    )
    return parser.parse_args()


def code_size_label(value: Any) -> str:
    return f"n={value_label(value)}"


def distance_label(value: Any) -> str:
    return f"d={value_label(value)}"


def code_name_label(value: Any) -> str:
    return value_label(value)


def qldlc_n_from_name(name: str) -> int | None:
    match = re.search(r"_n_(\d+)_", name)
    return int(match.group(1)) if match else None


QLDLC_CONFIG_KEYS = [
    "decoder",
    "schedule",
    "local_search",
    "sphere_decoding",
    "local_search_order",
    "search_radius",
    "full_basis",
]


def qldlc_config_tuple(record: Record) -> tuple[Any, ...]:
    return config_tuple(record, QLDLC_CONFIG_KEYS)


def qldlc_config_label(config: tuple[Any, ...]) -> str:
    pieces = []
    for key, value in zip(QLDLC_CONFIG_KEYS, config):
        if key == "full_basis" and value is False:
            continue
        if key == "search_radius" and value == 1.0:
            continue
        if key == "local_search_order" and value in ((1,), (1, 1, 1, 1, 1, 1, 1, 1)):
            pieces.append(f"order={len(value)}")
            continue
        pieces.append(kv_label(key, value))
    return ", ".join(pieces)


def qldlc_config_slug(config: tuple[Any, ...]) -> str:
    return safe_filename(qldlc_config_label(config))


def plot_qldlc_catalogue(input_csv: Path, output_dir: Path) -> None:
    configure_matplotlib()
    records = read_records(input_csv)
    output_dir = ensure_output_dir(output_dir)

    by_config: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    for record in records:
        by_config[qldlc_config_tuple(record)].append(record)

    summary_rows = []
    ranking_rows = []
    combined_pdf = output_dir / "qldlc_generated_instance_catalogue.pdf"

    sorted_configs = sorted(by_config.items(), key=lambda item: tuple(sort_key(v) for v in item[0]))
    with PdfPages(combined_pdf) as instance_pdf:
        for config_index, (config, config_records) in enumerate(sorted_configs, start=1):
            config_label = qldlc_config_label(config)
            slug = f"config_{config_index:03d}_{qldlc_config_slug(config)}"

            by_nbits: dict[int, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
            for record in config_records:
                nbits = int(record.metadata["nbits"])
                by_nbits[nbits][str(record.metadata["code_name"])].append(record)

            fig, axes = plt.subplots(
                len(by_nbits),
                1,
                figsize=(7.6, max(2.35 * len(by_nbits), 3.2)),
                sharex=True,
                constrained_layout=True,
            )
            if len(by_nbits) == 1:
                axes = [axes]

            for ax, (nbits, code_groups) in zip(axes, sorted(by_nbits.items())):
                sigma_values = sorted({round(r.sigma_scaled, 12) for group in code_groups.values() for r in group})
                rates_by_sigma: dict[float, list[float]] = {sigma: [] for sigma in sigma_values}

                for code_name, group_records in sorted(code_groups.items()):
                    group_records = sorted(group_records, key=lambda r: r.sigma_scaled)
                    xs = np.array([r.sigma_scaled for r in group_records])
                    ys = np.array([r.plot_rate for r in group_records])
                    for r in group_records:
                        rates_by_sigma[round(r.sigma_scaled, 12)].append(r.plot_rate)
                    ax.plot(xs, ys, color="#8b929a", alpha=0.18, linewidth=0.75)

                    ranking_rows.append(
                        {
                            "config_id": f"{config_index:03d}",
                            "config": config_label,
                            "nbits": nbits,
                            "code_name": code_name,
                            "points": len(group_records),
                            "sigma_min": f"{float(xs.min()):.6g}",
                            "sigma_max": f"{float(xs.max()):.6g}",
                            "min_error_rate": f"{float(np.min(ys)):.6g}",
                            "max_error_rate": f"{float(np.max(ys)):.6g}",
                            "mean_log10_error_rate": f"{float(np.mean(np.log10(np.clip(ys, 1e-12, 1.0)))):.6g}",
                        }
                    )

                xs = np.array(sigma_values)
                medians = np.array([np.median(rates_by_sigma[sigma]) for sigma in sigma_values])
                q10 = np.array([np.quantile(rates_by_sigma[sigma], 0.10) for sigma in sigma_values])
                q25 = np.array([np.quantile(rates_by_sigma[sigma], 0.25) for sigma in sigma_values])
                q75 = np.array([np.quantile(rates_by_sigma[sigma], 0.75) for sigma in sigma_values])
                q90 = np.array([np.quantile(rates_by_sigma[sigma], 0.90) for sigma in sigma_values])

                ax.fill_between(xs, q10, q90, color=PALETTE[0], alpha=0.10, linewidth=0, label="10-90%" if ax is axes[0] else None)
                ax.fill_between(xs, q25, q75, color=PALETTE[0], alpha=0.22, linewidth=0, label="25-75%" if ax is axes[0] else None)
                ax.plot(xs, medians, color=PALETTE[0], marker="o", markersize=3.8, linewidth=1.7, label="median" if ax is axes[0] else None)
                ax.set_yscale("log")
                ax.set_ylim(bottom=min(3e-5, np.nanmin(q10) * 0.6), top=1.05)
                ax.set_ylabel("Logical error rate")
                ax.set_title(f"nbits={nbits}, {len(code_groups)} instances", loc="left")

                for sigma in sigma_values:
                    vals = rates_by_sigma[sigma]
                    summary_rows.append(
                        {
                            "config_id": f"{config_index:03d}",
                            "config": config_label,
                            "nbits": nbits,
                            "sigma_scaled": f"{sigma:.6g}",
                            "codes": len(vals),
                            "median_error_rate": f"{float(np.median(vals)):.6g}",
                            "q10_error_rate": f"{float(np.quantile(vals, 0.10)):.6g}",
                            "q25_error_rate": f"{float(np.quantile(vals, 0.25)):.6g}",
                            "q75_error_rate": f"{float(np.quantile(vals, 0.75)):.6g}",
                            "q90_error_rate": f"{float(np.quantile(vals, 0.90)):.6g}",
                            "min_error_rate": f"{float(np.min(vals)):.6g}",
                            "max_error_rate": f"{float(np.max(vals)):.6g}",
                        }
                    )

            axes[0].legend(frameon=False, loc="upper left", ncols=3)
            axes[0].text(1.0, 1.08, config_label, transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8, alpha=0.82)
            axes[-1].set_xlabel(r"Noise strength $\sqrt{2\pi}\sigma$")
            fig.suptitle(f"Generated qLDLC overview - config {config_index:03d}", x=0.01, ha="left")
            fig.savefig(output_dir / f"{slug}_overview_by_nbits.png", bbox_inches="tight")
            fig.savefig(output_dir / f"{slug}_overview_by_nbits.pdf", bbox_inches="tight")
            plt.close(fig)

            records_by_code_n: dict[int, list[Record]] = defaultdict(list)
            for record in config_records:
                code_n = qldlc_n_from_name(str(record.metadata["code_name"])) or int(record.metadata["nbits"])
                records_by_code_n[code_n].append(record)

            for code_n, code_n_records in sorted(records_by_code_n.items()):
                fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
                by_code: dict[str, list[Record]] = defaultdict(list)
                for record in code_n_records:
                    by_code[str(record.metadata["code_name"])].append(record)

                for idx, (code_name, group_records) in enumerate(sorted(by_code.items())):
                    group_records = sorted(group_records, key=lambda r: r.sigma_scaled)
                    xs = np.array([r.sigma_scaled for r in group_records])
                    ys = np.array([r.plot_rate for r in group_records])
                    ax.plot(xs, ys, marker=".", markersize=3.2, linewidth=0.85, alpha=0.64, color=PALETTE[idx % len(PALETTE)], label=code_name.replace("reduced_ldlc_gkp_", ""))

                ax.set_title(f"qLDLC instances, config {config_index:03d}, code-name n={code_n}", loc="left")
                ax.text(1.0, 1.02, config_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.82)
                ax.set_xlabel(r"Noise strength $\sqrt{2\pi}\sigma$")
                ax.set_ylabel("Logical error rate")
                ax.set_yscale("log")
                ax.set_ylim(bottom=min(3e-5, min(r.plot_rate for r in code_n_records) * 0.6), top=1.05)
                if len(by_code) <= 30:
                    ax.legend(frameon=False, ncols=3, fontsize=6)
                instance_pdf.savefig(fig, bbox_inches="tight")
                fig.savefig(output_dir / f"{slug}_instances_n_{code_n}.png", bbox_inches="tight")
                plt.close(fig)

    if summary_rows:
        with (output_dir / "qldlc_summary_by_config_and_nbits.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    if ranking_rows:
        ranking_rows = sorted(ranking_rows, key=lambda row: (row["config_id"], int(row["nbits"]), float(row["mean_log10_error_rate"])))
        with (output_dir / "qldlc_instance_ranking.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ranking_rows[0]))
            writer.writeheader()
            writer.writerows(ranking_rows)

    config_rows = []
    for config_index, (config, config_records) in enumerate(sorted_configs, start=1):
        row = {
            "config_id": f"{config_index:03d}",
            "records": len(config_records),
            "codes": len({str(r.metadata["code_name"]) for r in config_records}),
            "config": qldlc_config_label(config),
        }
        row.update({key: value_label(value) for key, value in zip(QLDLC_CONFIG_KEYS, config)})
        config_rows.append(row)

    if config_rows:
        with (output_dir / "qldlc_configuration_index.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(config_rows[0]))
            writer.writeheader()
            writer.writerows(config_rows)

    print(f"Wrote {len(by_config)} qLDLC configuration overviews to {output_dir}")
    print(f"Wrote combined instance catalogue: {combined_pdf}")
