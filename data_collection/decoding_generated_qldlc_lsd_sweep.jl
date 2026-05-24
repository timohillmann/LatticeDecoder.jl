using Distributed

include(joinpath(@__DIR__, "decoding_generated_qldlc_overcomplete.jl"))

function qldlc_lsd_sweep_code_names()
    return sweep_code_names([
        "reduced_ldlc_gkp_n_13_1",
        "reduced_ldlc_gkp_n_14_1",
        "reduced_ldlc_gkp_n_15_1",
        "reduced_ldlc_gkp_n_16_1",
        "reduced_ldlc_gkp_n_17_1",
    ])
end

function qldlc_lsd_sweep_param_ranges()
    return Dict(
        :search_radius => [1.0],
        :decoder => ["lsd"],
        :schedule => ["serial"],
        :lsd_beta => sweep_betas(),
        :lsd_w_min => sweep_w_mins(),
        :sigmas => [[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6] ./ sqrt(2π)],
        :local_search => [true],
        :iterations => [50],
        :local_search_order => [[1, 1, 1, 1, 1, 1, 1, 1]],
        :local_search_lll => [false],
        :sphere_decoding => [false, true],
        :full_basis => [false],
    )
end

function main()
    path = sweep_output_path("results/qldlc/generated_qldlc_lsd_sweep.csv")
    code_names = qldlc_lsd_sweep_code_names()
    param_ranges = qldlc_lsd_sweep_param_ranges()

    if sweep_dry_run()
        println("qLDLC LSD sweep dry run")
        println("  output: $path")
        println("  codes: $(code_names)")
        println("  grid points before sigma expansion: $(length(parameter_grid(param_ranges)))")
        return nothing
    end

    run_qldlc_experiment!(
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
