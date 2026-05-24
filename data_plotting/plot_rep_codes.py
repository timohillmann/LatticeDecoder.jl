#!/usr/bin/env python3
"""Catalogue plots for repetition-code simulations."""

from overview_plotting import code_size_label, parse_common_args, plot_family_catalogue


def main() -> None:
    args = parse_common_args(
        default_input="results/rep_code/rep_code_decoding.csv",
        default_output="results/figures/rep_code",
        description="Plot repetition-code simulation configurations.",
    )
    plot_family_catalogue(
        input_csv=args.input,
        output_dir=args.output_dir,
        family_name="rep_code",
        group_key="n",
        group_label=code_size_label,
        config_keys=[
            "decoder",
            "schedule",
            "balance_mode",
            "local_search",
            "sphere_decoding",
            "search_radius",
        ],
        title_prefix="Repetition code",
    )


if __name__ == "__main__":
    main()
