using Distributed

include(joinpath(@__DIR__, "decoding_bivariate_bicycle_css.jl"))

function bivariate_bicycle_lsd_sweep_code_names()
    return sweep_code_names([
        "30_4_5_p2",
        "48_4_7_p2",
        "30_8_4_p2",
        "62_10_6_p2",
        "126_12_10_p2",
        "254_14_16_p2",
        # "510_16_24_p2",
    ])
end

function bivariate_bicycle_lsd_sweep_param_ranges()
    return Dict(
        :search_radius => [1.0],
        :decoder => ["lsd"],
        :schedule => ["serial"],
        :lsd_beta => sweep_betas(),
        :lsd_w_min => sweep_w_mins(),
        :sigmas => [collect(0.2:0.05:0.5) ./ sqrt(2π)],
        :basis => ["X"],
        :iterations => [50],
        :balance_weights => [true, false],
        :reduced_basis => [true],
        :local_search => [false],
        :local_search_lll => [false],
        :sphere_decoding => [false],
        :full_basis => [false],
    )
end

function main()
    path = sweep_output_path("results/bivariate_bicycle/bivariate_bicycle_lsd_sweep.csv")
    code_names = bivariate_bicycle_lsd_sweep_code_names()
    param_ranges = bivariate_bicycle_lsd_sweep_param_ranges()

    if sweep_dry_run()
        println("Bivariate bicycle LSD sweep dry run")
        println("  output: $path")
        println("  codes: $(code_names)")
        println("  grid points before sigma expansion: $(length(parameter_grid(param_ranges)))")
        return nothing
    end

    run_bivariate_bicycle_css_experiment!(
        path;
        code_names,
        param_ranges,
        n_samples = sweep_samples(),
        repeats = sweep_repeats(),
        max_errors = sweep_max_errors(),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
