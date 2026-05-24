#!/usr/bin/env python3
"""Catalogue plots for classical LDLC simulations."""

import math

from overview_plotting import parse_common_args, plot_family_catalogue, value_label


def snr_db_from_capacity(record) -> float:
    sigma = float(record.metadata["sigma"])
    snr_db = -10.0 * math.log10(2.0 * math.pi * math.e * sigma * sigma)
    return 0.0 if abs(snr_db) < 1e-12 else snr_db


def classical_ldlc_label(value) -> str:
    return value_label(value).replace("ldlc_", "")


def main() -> None:
    args = parse_common_args(
        default_input="results/classical_ldlc/classical_ldlc_decoding.csv",
        default_output="results/figures/classical_ldlc",
        description="Plot classical LDLC simulation configurations.",
    )
    plot_family_catalogue(
        input_csv=args.input,
        output_dir=args.output_dir,
        family_name="classical_ldlc",
        group_key="code_name",
        group_label=classical_ldlc_label,
        config_keys=[
            "decoder",
            "schedule",
            "d",
            "local_search",
            "sphere_decoding",
            "local_search_width",
            "local_search_order",
            "search_radius",
            "iterations",
        ],
        title_prefix="Classical LDLC",
        x_label="SNR from capacity (dB)",
        y_label="Symbol error rate",
        x_value_func=snr_db_from_capacity,
        x_index_label="snr_db",
    )


if __name__ == "__main__":
    main()
