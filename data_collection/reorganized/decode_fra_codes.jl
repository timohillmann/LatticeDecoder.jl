include("decode_generated_codes.jl")
include("load_fra_codes.jl")

const DEFAULT_FRA_RESULTS_PATH = "results/reorganized_codes/fra_ldlc_decoding.csv"

function parse_fra_stem(stem::String)
    m = match(r"reduced_ldlc_gkp_n_(\d+)_(\d+)$", stem)
    m === nothing && return nothing
    return (
        n = parse(Int, m.captures[1]),
        code_index = parse(Int, m.captures[2]),
    )
end

function discover_fra_artifacts(
    source_root::String = DEFAULT_FRA_CODES_PATH;
    start_code_index::Int = 1,
    end_code_index::Union{Nothing,Int} = nothing,
    max_codes::Union{Nothing,Int} = nothing,
)
    files = fra_code_files(source_root)
    artifacts = Dict{Symbol,Any}[]

    for path in files
        stem = splitext(basename(path))[1]
        parsed = parse_fra_stem(stem)
        parsed === nothing && error("Could not parse FRA filename: $(path)")
        push!(
            artifacts,
            Dict{Symbol,Any}(
                :jld2_path => path,
                :n => parsed.n,
                :task_id => "fra",
                :attempt => 1,
                :code_index => parsed.code_index,
                :code_name => stem,
                :generator_key => "qubit_generator_lll",
            ),
        )
    end

    sort!(artifacts, by = a -> (a[:n], a[:code_index], a[:code_name]))

    total_codes = length(artifacts)
    total_codes > 0 || return artifacts

    start_code_index >= 1 || error("start_code_index must be >= 1.")
    start_code_index <= total_codes || error("start_code_index=$(start_code_index) exceeds available FRA codes ($(total_codes)).")

    if end_code_index !== nothing
        end_code_index >= start_code_index || error("end_code_index must be >= start_code_index.")
        end_code_index <= total_codes || error("end_code_index=$(end_code_index) exceeds available FRA codes ($(total_codes)).")
    end

    for (idx, artifact) in enumerate(artifacts)
        artifact[:global_artifact_index] = idx
    end

    final_end_index = end_code_index === nothing ? total_codes : end_code_index
    artifacts = artifacts[start_code_index:final_end_index]

    if max_codes !== nothing
        max_codes > 0 || error("max_codes must be positive when provided.")
        max_codes < length(artifacts) && (artifacts = artifacts[1:max_codes])
    end

    return artifacts
end

function load_fra_generated_code(artifact::Dict{Symbol,Any})
    code = load_fra_code(artifact[:jld2_path])
    generator_key = artifact[:generator_key]
    haskey(code, generator_key) || error("FRA code $(artifact[:jld2_path]) has no key $(generator_key).")

    G = Float64.(Matrix(code[generator_key]))
    size(G, 1) == size(G, 2) || error("Expected square FRA generator for $(artifact[:jld2_path]).")

    H = inv(G)
    inv_residual = norm(H * G - I(size(G, 1)))
    return H, G, inv_residual
end

function run_fra_code_experiment(
    p::Dict{Symbol,Any},
    artifact::Dict{Symbol,Any},
    n_samples::Int64,
)
    H, G, inv_residual = load_fra_generated_code(artifact)
    order = p[:local_search] ? Int64[2; fill(1, max(artifact[:n] - 1, 0))] : Int64[0]
    p[:iterations] = size(H, 2)

    results = qec_experiment(;
        H = H,
        G = G,
        order = order,
        n_samples = n_samples,
        params = p,
    )

    return results, H, inv_residual
end

function run_fra_codes_decoding(;
    source_root::String = DEFAULT_FRA_CODES_PATH,
    results_path::String = DEFAULT_FRA_RESULTS_PATH,
    start_code_index::Int = 103,
    end_code_index::Union{Nothing,Int} = nothing,
    max_codes::Union{Nothing,Int} = nothing,
    n_samples::Int = 250,
    repeats::Int = 1,
    target_workers::Int = 10,
    sigmas::Vector{Float64} = Float64[0.35, 0.4] ./ sqrt(2π),
    decoder::String = "lsd",
    schedule::String = "serial",
    decoding_style::String = "received_vector",
    search_radius::Float64 = 1.0,
    local_search::Bool = true,
    local_search_lll::Bool = false,
    sphere_decoding::Bool = false,
    full_basis::Bool = false,
)
    ensure_worker_count(target_workers)

    isempty(sigmas) && error("sigmas must contain at least one value.")
    n_samples > 0 || error("n_samples must be positive.")
    repeats > 0 || error("repeats must be positive.")

    artifacts = discover_fra_artifacts(
        source_root;
        start_code_index = start_code_index,
        end_code_index = end_code_index,
        max_codes = max_codes,
    )
    isempty(artifacts) && error("No FRA artifacts found.")

    param_ranges = Dict(
        :search_radius => [search_radius],
        :decoder => [decoder],
        :local_search => [local_search],
        :local_search_lll => [local_search_lll],
        :schedule => [schedule],
        :decoding_style => [decoding_style],
        :sigmas => [sigmas],
        :sphere_decoding => [sphere_decoding],
        :full_basis => [full_basis],
    )

    keys_list = collect(keys(param_ranges))
    values_list = collect(values(param_ranges))
    params_list = [
        Dict{Symbol,Any}(zip(keys_list, combo))
        for combo in product(values_list...)
    ]

    total_runs = repeats * length(artifacts) * length(params_list)
    run_index = 1

    first_idx = first(artifacts)[:global_artifact_index]
    last_idx = last(artifacts)[:global_artifact_index]
    println("Decoding $(length(artifacts)) FRA code(s) from $(source_root)")
    println("Artifact slice: [$(first_idx):$(last_idx)] in sorted artifact order")
    println("Writing simulation data to: $(results_path)")
    println("n_samples=$(n_samples), repeats=$(repeats), settings=$(length(params_list))")

    for rep in 1:repeats
        for artifact in artifacts
            for p_base in params_list
                p = copy(p_base)
                p[:n] = artifact[:n]
                p[:code_name] = artifact[:code_name]
                p[:task_id] = artifact[:task_id]
                p[:attempt] = artifact[:attempt]
                p[:code_index] = artifact[:code_index]
                p[:global_artifact_index] = artifact[:global_artifact_index]

                println("Run $(run_index) / $(total_runs): $(artifact[:code_name])")
                run_index += 1

                try
                    results, H, inv_residual = run_fra_code_experiment(p, artifact, n_samples)

                    for (σ, res) in zip(p[:sigmas], results)
                        meta = build_metadata(
                            p,
                            σ,
                            H;
                            repeat = rep,
                            code_file = artifact[:jld2_path],
                            source_type = "fra_jld2",
                            generator_key = artifact[:generator_key],
                            matrix_inverse_residual = inv_residual,
                        )

                        add_data!(
                            results_path;
                            shots = n_samples,
                            errors = res,
                            decoder = string(p[:decoder]),
                            json_metadata = meta,
                        )
                    end
                catch err
                    @warn "Skipping failed FRA code experiment." code_file = artifact[:jld2_path] exception = (err, catch_backtrace())
                end

                flush(stdout)
            end
        end
    end
end

"""
53, 70, 93, 97, 100, 103, 104, 105, 106, 107,
108, 109, 110, 111, 112, 113, 114, 115, 116,
117, 118, 119, 120, 121, 122, 123, 124, 125,
126, 127
"""

const FRA_RUN_SETTINGS = (
    source_root = DEFAULT_FRA_CODES_PATH,
    results_path = DEFAULT_FRA_RESULTS_PATH,
    start_code_index = 103,
    end_code_index = nothing,
    max_codes = nothing,
    n_samples = 100,
    repeats = 1,
    target_workers = 4,
    sigmas = [0.3, 0.4],
    sigmas_already_normalized = false,
    decoder = "nearest",
    schedule = "serial",
    decoding_style = "received_vector",
    search_radius = 1.0,
    local_search = true,
    local_search_lll = false,
    sphere_decoding = false,
    full_basis = false,
)

function run_fra_with_script_settings(settings = FRA_RUN_SETTINGS)
    sigmas = normalize_sigmas(
        settings.sigmas;
        already_normalized = settings.sigmas_already_normalized,
    )

    run_fra_codes_decoding(
        source_root = settings.source_root,
        results_path = settings.results_path,
        start_code_index = settings.start_code_index,
        end_code_index = settings.end_code_index,
        max_codes = settings.max_codes,
        n_samples = settings.n_samples,
        repeats = settings.repeats,
        target_workers = settings.target_workers,
        sigmas = sigmas,
        decoder = settings.decoder,
        schedule = settings.schedule,
        decoding_style = settings.decoding_style,
        search_radius = settings.search_radius,
        local_search = settings.local_search,
        local_search_lll = settings.local_search_lll,
        sphere_decoding = settings.sphere_decoding,
        full_basis = settings.full_basis,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_fra_with_script_settings()
end
