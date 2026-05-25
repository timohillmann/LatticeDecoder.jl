using Distributed

include(joinpath(@__DIR__, "decoding_bivariate_bicycle_css.jl"))

function bivariate_bicycle_memory_sweep_code_names()
    return memory_sweep_code_names([
        "30_4_5_p2",
        "48_4_7_p2",
        "30_8_4_p2",
        "62_10_6_p2",
        "126_12_10_p2",
        "254_14_16_p2",
        # "510_16_24_p2",
    ])
end

memory_sweep_samples(default::Int = 1_000) =
    _positive_int_env("MEMORY_SWEEP_SAMPLES", default)

memory_sweep_repeats(default::Int = 250) =
    _positive_int_env("MEMORY_SWEEP_REPEATS", default)

function memory_sweep_max_errors(default::Int = 200)
    value = _nonnegative_int_env("MEMORY_SWEEP_MAX_ERRORS")
    return value === nothing ? default : value
end

memory_sweep_strengths(default::Vector{Float64} = collect(-1.0:0.1:1.0)) =
    _float_list_env("MEMORY_SWEEP_STRENGTHS", default)

memory_sweep_betas(default::Vector{Float64} = collect(1.5:0.1:6.0)) =
    _float_list_env("MEMORY_SWEEP_BETAS", default)

memory_sweep_w_mins(default::Vector{Float64} = collect(0.6:0.05:1.15)) =
    _float_list_env("MEMORY_SWEEP_W_MINS", default)

memory_sweep_code_names(default::Vector{String}) =
    _string_list_env("MEMORY_SWEEP_CODE_NAMES", default)

memory_sweep_output_path(default::AbstractString) =
    get(ENV, "MEMORY_SWEEP_OUTPUT_PATH", default)

memory_sweep_dry_run(default::Bool = false) =
    _bool_env("MEMORY_SWEEP_DRY_RUN", default)

function bivariate_bicycle_memory_sweep_param_ranges()
    return Dict(
        :search_radius => [1.0],
        :decoder => ["lsd"],
        :schedule => ["serial"],
        :lsd_beta => memory_sweep_betas(),
        :lsd_w_min => memory_sweep_w_mins(),
        :memory_strength => memory_sweep_strengths(),
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
    path = memory_sweep_output_path("results/bivariate_bicycle/bivariate_bicycle_memory_sweep.csv")
    code_names = bivariate_bicycle_memory_sweep_code_names()
    param_ranges = bivariate_bicycle_memory_sweep_param_ranges()

    if memory_sweep_dry_run()
        println("Bivariate bicycle memory sweep dry run")
        println("  output: $path")
        println("  codes: $(code_names)")
        println("  memory strengths: $(param_ranges[:memory_strength])")
        println("  LSD betas: $(param_ranges[:lsd_beta])")
        println("  LSD w_mins: $(param_ranges[:lsd_w_min])")
        println("  grid points before sigma expansion: $(length(parameter_grid(param_ranges)))")
        return nothing
    end

    println("Using $(nworkers()) workers for the sweep")

    run_bivariate_bicycle_css_experiment!(
        path;
        code_names,
        param_ranges,
        n_samples = memory_sweep_samples(),
        repeats = memory_sweep_repeats(),
        max_errors = memory_sweep_max_errors(),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
