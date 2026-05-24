#!/usr/bin/env python3
"""Catalogue plots for non-CSS surface-code simulations."""

from overview_plotting import distance_label, parse_common_args, plot_family_catalogue


def main() -> None:
    args = parse_common_args(
        default_input="results/surface_code/surface_code_noncss_decoding.csv",
        default_output="results/figures/surface_code_noncss",
        description="Plot non-CSS surface-code simulation configurations.",
    )
    plot_family_catalogue(
        input_csv=args.input,
        output_dir=args.output_dir,
        family_name="surface_code_noncss",
        group_key="d",
        group_label=distance_label,
        config_keys=[
            "decoder",
            "schedule",
            "local_search",
            "sphere_decoding",
            "full_basis",
            "css_decoding",
            "balance_hamming_weight",
            "bit_flip",
        ],
        xlim_l=0.3,
        xlim_r=0.8,
        title_prefix="Surface code non-CSS",
    )


if __name__ == "__main__":
    main()
