#!/usr/bin/env python3
"""Catalogue plots for bivariate-bicycle and related CSS-code simulations."""

from overview_plotting import code_name_label, parse_common_args, plot_family_catalogue


def main() -> None:
    args = parse_common_args(
        default_input="results/bivariate_bicycle/bivariate_bicycle_css_decoding.csv",
        default_output="results/figures/bivariate_bicycle",
        description="Plot bivariate-bicycle CSS simulation configurations.",
    )
    plot_family_catalogue(
        input_csv=args.input,
        output_dir=args.output_dir,
        family_name="bivariate_bicycle_css",
        group_key="code_name",
        group_label=code_name_label,
        config_keys=[
            "decoder",
            "schedule",
            "code_family",
            "basis",
            "local_search",
            "sphere_decoding",
            "balance_weights",
            "reduced_basis",
            "search_radius",
        ],
        title_prefix="Bivariate bicycle / CSS family",
    )


if __name__ == "__main__":
    main()
