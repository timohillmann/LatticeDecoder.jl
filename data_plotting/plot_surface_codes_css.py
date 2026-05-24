#!/usr/bin/env python3
"""Catalogue plots for CSS surface-code simulations."""

from overview_plotting import distance_label, parse_common_args, plot_family_catalogue


def main() -> None:
    args = parse_common_args(
        default_input="results/surface_code/surface_code_css_decoding.csv",
        default_output="results/figures/surface_code_css",
        description="Plot CSS surface-code simulation configurations.",
    )
    plot_family_catalogue(
        input_csv=args.input,
        output_dir=args.output_dir,
        family_name="surface_code_css",
        group_key="d",
        group_label=distance_label,
        config_keys=[
            "decoder",
            "schedule",
            "basis",
            "local_search",
            "sphere_decoding",
            "balance_hamming_weight",
            "search_radius",
        ],
        title_prefix="Surface code CSS",
    )


if __name__ == "__main__":
    main()
