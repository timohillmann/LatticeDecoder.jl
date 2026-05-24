#!/usr/bin/env python3
"""Overview plots for generated quantum-LDLC simulations."""

from overview_plotting import parse_common_args, plot_qldlc_catalogue


def main() -> None:
    args = parse_common_args(
        default_input="results/qldlc/generated_qldlc_overcomplete.csv",
        default_output="results/figures/qldlc_generated",
        description="Plot generated quantum-LDLC simulation overviews.",
    )
    plot_qldlc_catalogue(args.input, args.output_dir)


if __name__ == "__main__":
    main()
