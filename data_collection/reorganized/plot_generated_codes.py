from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sinter


# Script-first settings: edit these values directly.
ROOT = Path("/Users/timo/Documents/LatticeDecoder.jl")
CSV_PATH = ROOT / "results/reorganized_codes/reduced_ldlc_generated_decoding.csv"
FIG_DIR = ROOT / "results/figures/reorganized_codes"

DECODER = "lsd"
SCHEDULE = "serial"
LOCAL_SEARCH = True
LOCAL_SEARCH_LLL = False
SPHERE_DECODING = False

# If None, uses all n values present in the file.
INCLUDE_NS: Sequence[int] | None = None


# For per-code plots, show up to this many codes for the selected n.
MAX_CODES_PER_N = 10

# Scaling plot will group by nearest sigma in this list (physical sigma).
SCALING_SIGMAS = [0.35, 0.4, 0.45, 0.5]
SCALING_SIGMA_TOL = 0.015


MARKERS = [
    "o",
    "s",
    "d",
    "p",
    "X",
    "^",
    "v",
    "<",
    ">",
    "h",
    "D",
    "P",
    "x",
] * 30

COLORS = list(plt.get_cmap("tab20").colors) * 30


def lighten_color(color, amount: float = 0.5):
    c = np.array(mcolors.to_rgb(color))
    return tuple((1.0 - amount) * c + amount * np.array([1.0, 1.0, 1.0]))


def marker_style(index: int, *_args):
    color = COLORS[index % len(COLORS)]
    marker = MARKERS[index % len(MARKERS)]
    return dict(
        markerfacecolor=lighten_color(color, 0.45),
        markeredgecolor=color,
        color=color,
        markersize=4.5,
        linestyle="solid",
        marker=marker,
    )


def sigma_phys(stat) -> float:
    return float(np.sqrt(2 * np.pi) * stat.json_metadata["sigma"])


def matches_common_filters(
    stat,
    *,
    decoder: str,
    schedule: str,
    local_search: bool,
    local_search_lll: bool,
    sphere_decoding: bool,
) -> bool:
    md = stat.json_metadata
    return (
        md.get("decoder") == decoder
        and md.get("schedule") == schedule
        and md.get("local_search") == local_search
        and md.get("local_search_lll") == local_search_lll
        and md.get("sphere_decoding") == sphere_decoding
    )


def nearest_target_sigma(
    sigma_value: float,
    targets: Sequence[float],
    tol: float,
) -> float | None:
    if len(targets) == 0:
        return None
    arr = np.asarray(targets, dtype=float)
    idx = int(np.argmin(np.abs(arr - sigma_value)))
    best = float(arr[idx])
    if abs(best - sigma_value) <= tol:
        return best
    return None


def save_figure(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_threshold_by_n(
    stats: Sequence,
    *,
    decoder: str,
    schedule: str,
    local_search: bool,
    local_search_lll: bool,
    sphere_decoding: bool,
    include_ns: Sequence[int] | None,
):
    filtered = [
        st
        for st in stats
        if matches_common_filters(
            st,
            decoder=decoder,
            schedule=schedule,
            local_search=local_search,
            local_search_lll=local_search_lll,
            sphere_decoding=sphere_decoding,
        )
        and (include_ns is None or int(st.json_metadata["n"]) in include_ns)
    ]
    if len(filtered) == 0:
        print("No matching stats for threshold-by-n plot.")
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    sinter.plot_error_rate(
        ax=ax,
        stats=filtered,
        x_func=lambda st: sigma_phys(st),
        group_func=lambda st: {"label": f"n = {int(st.json_metadata['n'])}", "sort": int(st.json_metadata["n"])},
        plot_args_func=marker_style,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"Noise strength $\sigma$")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Generated Reduced Codes: Threshold By n")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, ncols=2)

    stem = (
        f"{CSV_PATH.stem}_threshold_by_n_"
        f"{decoder}_{schedule}_ls{int(local_search)}_lll{int(local_search_lll)}_sd{int(sphere_decoding)}"
    )
    save_figure(fig, stem)


def plot_threshold_by_code(
    stats: Sequence,
    *,
    decoder: str,
    schedule: str,
    local_search: bool,
    local_search_lll: bool,
    sphere_decoding: bool,
    selected_ns: Sequence[int] | None,
    max_codes: int,
):
    base = [
        st
        for st in stats
        if matches_common_filters(
            st,
            decoder=decoder,
            schedule=schedule,
            local_search=local_search,
            local_search_lll=local_search_lll,
            sphere_decoding=sphere_decoding,
        )
    ]
    if len(base) == 0:
        print("No matching stats for threshold-by-code plot.")
        return

    ns = sorted({int(st.json_metadata["n"]) for st in base})
    if len(ns) == 0:
        print("No n values found for threshold-by-code plot.")
        return

    ns_to_plot = list(ns) if selected_ns is None else [n for n in selected_ns if n in ns]
    if len(ns_to_plot) == 0:
        print("No selected n values are present in the dataset for threshold-by-code plot.")
        return

    for n_value in ns_to_plot:
        codes_for_n = sorted({st.json_metadata["code_name"] for st in base if int(st.json_metadata["n"]) == n_value})
        chosen_codes = set(codes_for_n[:max_codes])

        filtered = [st for st in base if int(st.json_metadata["n"]) == n_value and st.json_metadata["code_name"] in chosen_codes]
        if len(filtered) == 0:
            print(f"No stats available for n={n_value} in threshold-by-code plot.")
            continue

        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        sinter.plot_error_rate(
            ax=ax,
            stats=filtered,
            x_func=lambda st: sigma_phys(st),
            group_func=lambda st: st.json_metadata["code_name"],
            plot_args_func=marker_style,
        )
        ax.set_yscale("log")
        ax.set_xlabel(r"Noise strength $\sigma$")
        ax.set_ylabel("Logical error rate")
        ax.set_title(f"Generated Reduced Codes: Threshold By Code (n={n_value})")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=6, ncols=1)

        stem = (
            f"{CSV_PATH.stem}_threshold_by_code_n{n_value}_"
            f"{decoder}_{schedule}_ls{int(local_search)}"
        )
        save_figure(fig, stem)


def plot_scaling_vs_n(
    stats: Sequence,
    *,
    decoder: str,
    schedule: str,
    local_search: bool,
    local_search_lll: bool,
    sphere_decoding: bool,
    sigma_targets: Sequence[float],
    sigma_tol: float,
):
    filtered = []
    for st in stats:
        if not matches_common_filters(
            st,
            decoder=decoder,
            schedule=schedule,
            local_search=local_search,
            local_search_lll=local_search_lll,
            sphere_decoding=sphere_decoding,
        ):
            continue
        s_target = nearest_target_sigma(sigma_phys(st), sigma_targets, sigma_tol)
        if s_target is None:
            continue
        filtered.append(st)

    if len(filtered) == 0:
        print("No matching stats for scaling-vs-n plot.")
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    def _group(st):
        target = nearest_target_sigma(sigma_phys(st), sigma_targets, sigma_tol)
        assert target is not None
        return {"label": rf"$\sigma$ = {target:.2f}", "sort": target}

    sinter.plot_error_rate(
        ax=ax,
        stats=filtered,
        x_func=lambda st: int(st.json_metadata["n"]),
        group_func=_group,
        plot_args_func=marker_style,
    )

    n_values = sorted({int(st.json_metadata["n"]) for st in filtered})
    ax.set_yscale("log")
    ax.set_xlabel("Mode count n")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Generated Reduced Codes: Scaling vs n")
    if len(n_values) > 0:
        ax.set_xticks(n_values)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    stem = (
        f"{CSV_PATH.stem}_scaling_vs_n_"
        f"{decoder}_{schedule}_ls{int(local_search)}"
    )
    save_figure(fig, stem)


def print_dataset_summary(stats: Iterable):
    stats = list(stats)
    ns = sorted({int(st.json_metadata["n"]) for st in stats})
    decoders = sorted({str(st.json_metadata.get("decoder")) for st in stats})
    schedules = sorted({str(st.json_metadata.get("schedule")) for st in stats})
    sigma_values = sorted({round(float(sigma_phys(st)), 6) for st in stats})

    print(f"Loaded {len(stats)} stat rows from: {CSV_PATH}")
    print(f"n values: {ns}")
    print(f"decoders: {decoders}")
    print(f"schedules: {schedules}")
    print(f"physical sigmas: {sigma_values}")


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

    stats = list(sinter.stats_from_csv_files(str(CSV_PATH)))
    if len(stats) == 0:
        raise RuntimeError(f"No stats found in: {CSV_PATH}")

    print_dataset_summary(stats)

    plot_threshold_by_n(
        stats,
        decoder=DECODER,
        schedule=SCHEDULE,
        local_search=LOCAL_SEARCH,
        local_search_lll=LOCAL_SEARCH_LLL,
        sphere_decoding=SPHERE_DECODING,
        include_ns=INCLUDE_NS,
    )

    plot_threshold_by_code(
        stats,
        decoder=DECODER,
        schedule=SCHEDULE,
        local_search=LOCAL_SEARCH,
        local_search_lll=LOCAL_SEARCH_LLL,
        sphere_decoding=SPHERE_DECODING,
        selected_ns=INCLUDE_NS,
        max_codes=MAX_CODES_PER_N,
    )

    plot_scaling_vs_n(
        stats,
        decoder=DECODER,
        schedule=SCHEDULE,
        local_search=LOCAL_SEARCH,
        local_search_lll=LOCAL_SEARCH_LLL,
        sphere_decoding=SPHERE_DECODING,
        sigma_targets=SCALING_SIGMAS,
        sigma_tol=SCALING_SIGMA_TOL,
    )

    plt.show()


if __name__ == "__main__":
    main()
