using Distributed

using LatticeDecoder
using LinearAlgebra
using NPZ
using IterTools: product
using Serialization

function initialize_worker_context!()
    pids = workers()
    isempty(pids) && return nothing
    Distributed.remotecall_eval(
        Main,
        pids,   
        quote
            using LatticeDecoder
            using LinearAlgebra
            using NPZ
        end,
    )
    return nothing
end

function ensure_worker_count(target_workers::Int)
    workers_to_add = max(0, target_workers - nworkers())
    workers_to_add > 0 && addprocs(workers_to_add)
    initialize_worker_context!()
    return nothing
end

const DEFAULT_SOURCE_ROOT = "/Users/timo/Documents/gkp_ldlc_mwe-main/examples/reorganized"
const DEFAULT_RESULTS_PATH = "results/reorganized_codes/reduced_ldlc_generated_decoding.csv"

function qec_sample(;
    tg::LatticeDecoder.TannerGraph,
    lsd::LatticeDecoder.LocalSearch,
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical_check::AbstractMatrix{Float64},
    σ::Float64,
    n_samples::Int,
    schedule::String = "serial",
    iterations::Int = tg.nv,
    decoder::Union{String,Int64} = "lsd",
    decoding_style::String = "syndrome",
    search_radius::Float64 = 1.5,
    local_search::Bool = false,
    extras...,
)
    if schedule == "serial"
        run_bp! = run_serial_belief_propagation!
    else
        run_bp! = run_belief_propagation!
    end

    tot_errors = @distributed (+) for _ = 1:n_samples
        y = sample_error(σ, tg.nv)
        if decoding_style == "syndrome"
            s = (H * y) .% 1
            η = G * s
        elseif decoding_style == "received_vector"
            η = copy(y)
        else
            error("Invalid decoding style. Choose either 'syndrome' or 'received_vector'.")
        end

        bp_result = run_bp!(tg, η, σ, iterations, decoder, search_interval = search_radius)
        dec = hard_decision(bp_result, H)

        if local_search
            λ = abs.(H * bp_result) .% 1.0
            local_search!(η, λ, dec, lsd)
        end

        corr = η - lsd.G * dec
        res = y - corr

        log_ok = true
        log_check = logical_check' * res
        @inbounds for x in log_check
            if abs(x - round(x)) >= 1e-5
                log_ok = false
                break
            end
        end
        log_ok ? 0 : 1
    end

    return tot_errors
end

function qec_experiment(;
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    order::Vector{Int64},
    n_samples::Int64,
    params::Dict{Symbol,Any},
)
    tg = initialize_tanner_graph(H)
    lsd = LocalSearch(
        length(order),
        G,
        order,
        params[:local_search_lll],
        params[:sphere_decoding],
        params[:full_basis],
    )
    logical_check = G

    results = Int[]
    for σ in params[:sigmas]
        print("σ = $(σ)\r")
        push!(
            results,
            qec_sample(;
                H = H,
                G = G,
                logical_check = logical_check,
                lsd = lsd,
                tg = tg,
                σ = σ,
                n_samples = n_samples,
                params...,
            ),
        )
    end

    return results
end

function canonicalize_keys(data)::Dict{String,Any}
    out = Dict{String,Any}()
    for (k, v) in pairs(data)
        out[string(k)] = v
    end
    return out
end

function safe_to_int(x)::Union{Int,Nothing}
    try
        return Int(x)
    catch
        return nothing
    end
end

function parse_stem_tokens(stem::String)
    m = match(r"n_(\d+)_task_(.+?)_attempt_(\d+)_code_(\d+)_([0-9a-fA-F-]+)$", stem)
    m === nothing && return nothing
    return (
        n = parse(Int, m.captures[1]),
        task_id = m.captures[2],
        attempt = parse(Int, m.captures[3]),
        code_index = parse(Int, m.captures[4]),
        uuid = m.captures[5],
    )
end

function infer_npy_path(jls_path::String, prefix::String)::Union{String,Nothing}
    stem = splitext(basename(jls_path))[1]
    tokens = parse_stem_tokens(stem)
    tokens === nothing && return nothing

    file_prefix = "$(prefix)_n_$(tokens.n)_task_$(tokens.task_id)_attempt_$(tokens.attempt)_code_$(tokens.code_index)_"
    candidates = filter(
        p -> startswith(basename(p), file_prefix) && endswith(p, ".npy"),
        readdir(dirname(jls_path); join = true),
    )
    sort!(candidates)
    return isempty(candidates) ? nothing : first(candidates)
end

function resolve_artifact_path(
    data::Dict{String,Any},
    key::String,
    jls_path::String;
    fallback_prefix::String,
)::Union{String,Nothing}
    if haskey(data, key) && data[key] isa AbstractString
        raw = String(data[key])
        candidate = isabspath(raw) ? raw : normpath(joinpath(dirname(jls_path), raw))
        isfile(candidate) && return candidate
    end
    return infer_npy_path(jls_path, fallback_prefix)
end

function load_artifact_metadata(jls_path::String)::Union{Dict{Symbol,Any},Nothing}
    raw = deserialize(jls_path)
    raw isa Dict || error("Serialized payload must be a Dict: $(jls_path)")
    data = canonicalize_keys(raw)

    gen_path = resolve_artifact_path(
        data,
        "reduced_generator_path",
        jls_path;
        fallback_prefix = "reduced_generator",
    )
    inv_path = resolve_artifact_path(
        data,
        "reduced_generator_inv_path",
        jls_path;
        fallback_prefix = "reduced_generator_inv",
    )

    if gen_path === nothing || inv_path === nothing
        @warn "Skipping artifact with unresolved generator paths." jls_path gen_path inv_path
        return nothing
    end

    stem = splitext(basename(jls_path))[1]
    stem_tokens = parse_stem_tokens(stem)

    n = haskey(data, "n") ? safe_to_int(data["n"]) : nothing
    n === nothing && stem_tokens !== nothing && (n = stem_tokens.n)
    n === nothing && error("Could not infer n for artifact: $(jls_path)")

    task_id = haskey(data, "task_id") ? string(data["task_id"]) : (stem_tokens === nothing ? "unknown" : stem_tokens.task_id)
    attempt = haskey(data, "attempt") ? safe_to_int(data["attempt"]) : (stem_tokens === nothing ? nothing : stem_tokens.attempt)
    code_index = haskey(data, "code_index") ? safe_to_int(data["code_index"]) : (stem_tokens === nothing ? nothing : stem_tokens.code_index)

    code_name = "n$(n)_task_$(task_id)_attempt_$(attempt)_code_$(code_index)"

    return Dict{Symbol,Any}(
        :jls_path => jls_path,
        :generator_path => gen_path,
        :generator_inv_path => inv_path,
        :n => n,
        :task_id => task_id,
        :attempt => attempt,
        :code_index => code_index,
        :code_name => code_name,
    )
end

function candidate_input_dirs(
    source_root::String;
    reduced_codes_dir::Union{Nothing,String} = nothing,
)
    dirs = String[]
    push!(dirs, source_root)

    sibling = joinpath(dirname(source_root), "reduced_ldlc_gens")
    sibling != source_root && push!(dirs, sibling)

    if reduced_codes_dir !== nothing
        explicit_dir = strip(reduced_codes_dir)
        !isempty(explicit_dir) && push!(dirs, explicit_dir)
    end

    return unique(filter(isdir, dirs))
end

function discover_artifacts(
    source_root::String;
    reduced_codes_dir::Union{Nothing,String} = nothing,
    start_code_index::Int = 1,
    end_code_index::Union{Nothing,Int} = nothing,
    max_codes::Union{Nothing,Int} = nothing,
)
    input_dirs = candidate_input_dirs(source_root; reduced_codes_dir = reduced_codes_dir)
    isempty(input_dirs) && error("No valid input directory found from source root: $(source_root)")

    jls_files = String[]
    for dir in input_dirs
        for (root, _, files) in walkdir(dir)
            for file in files
                endswith(file, ".jls") && push!(jls_files, joinpath(root, file))
            end
        end
    end
    jls_files = unique(jls_files)
    sort!(jls_files)
    isempty(jls_files) && error("No .jls files found in: $(join(input_dirs, ", "))")

    artifacts = Dict{Symbol,Any}[]
    for jls_path in jls_files
        metadata = load_artifact_metadata(jls_path)
        metadata === nothing || push!(artifacts, metadata)
    end

    sort!(artifacts, by = a -> (a[:n], a[:code_name]))

    total_codes = length(artifacts)
    total_codes > 0 || return artifacts

    start_code_index >= 1 || error("start_code_index must be >= 1.")
    start_code_index <= total_codes || error("start_code_index=$(start_code_index) exceeds available codes ($(total_codes)).")

    if end_code_index !== nothing
        end_code_index >= start_code_index || error("end_code_index must be >= start_code_index.")
        end_code_index <= total_codes || error("end_code_index=$(end_code_index) exceeds available codes ($(total_codes)).")
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

function load_npy_matrix(path::String)::Matrix{Float64}
    data = npzread(path)
    data isa AbstractArray || error("Expected array data in $(path).")
    ndims(data) == 2 || error("Expected 2D matrix in $(path), got ndims=$(ndims(data)).")
    return Float64.(Matrix(data))
end

function load_generated_code(artifact::Dict{Symbol,Any})
    G = load_npy_matrix(artifact[:generator_path])
    H = load_npy_matrix(artifact[:generator_inv_path])
    size(H) == size(G) || error("H and G dimensions must match for $(artifact[:jls_path]).")
    size(H, 1) == size(H, 2) || error("Expected square H for $(artifact[:jls_path]).")
    inv_residual = norm(H * G - I(size(H, 1)))
    return H, G, inv_residual
end

function build_metadata(p, σ, H; extra...)
    meta = Dict{Symbol,Any}()
    for (k, v) in pairs(p)
        meta[k] = v
    end

    meta[:sigma] = σ
    meta[:nbits] = size(H, 2)

    for (k, v) in extra
        meta[k] = v
    end

    delete!(meta, :sigmas)

    if haskey(meta, :local_search) && !(meta[:local_search])
        meta[:local_search_lll] = false
        meta[:sphere_decoding] = false
    end

    return meta
end

function run_generated_code_experiment(
    p::Dict{Symbol,Any},
    artifact::Dict{Symbol,Any},
    n_samples::Int64,
)
    H, G, inv_residual = load_generated_code(artifact)
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

function run_generated_codes_decoding(;
    source_root::String = DEFAULT_SOURCE_ROOT,
    reduced_codes_dir::Union{Nothing,String} = nothing,
    results_path::String = DEFAULT_RESULTS_PATH,
    start_code_index::Int = 1,
    end_code_index::Union{Nothing,Int} = nothing,
    max_codes::Union{Nothing,Int} = nothing,
    n_samples::Int = 1_000,
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

    artifacts = discover_artifacts(
        source_root;
        reduced_codes_dir = reduced_codes_dir,
        start_code_index = start_code_index,
        end_code_index = end_code_index,
        max_codes = max_codes,
    )
    isempty(artifacts) && error("No decodable artifacts found.")

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
    println("Decoding $(length(artifacts)) code(s) from $(source_root)")
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
                    results, H, inv_residual = run_generated_code_experiment(p, artifact, n_samples)

                    for (σ, res) in zip(p[:sigmas], results)
                        meta = build_metadata(
                            p,
                            σ,
                            H;
                            repeat = rep,
                            code_file = artifact[:jls_path],
                            reduced_generator_path = artifact[:generator_path],
                            reduced_generator_inv_path = artifact[:generator_inv_path],
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
                    @warn "Skipping failed code experiment." code_file = artifact[:jls_path] exception = (err, catch_backtrace())
                end

                flush(stdout)
            end
        end
    end
end

function normalize_sigmas(sigmas::AbstractVector{<:Real}; already_normalized::Bool = false)
    out = Float64.(sigmas)
    return already_normalized ? out : out ./ sqrt(2π)
end

const RUN_SETTINGS = (
    source_root = DEFAULT_SOURCE_ROOT,
    reduced_codes_dir = nothing,
    results_path = DEFAULT_RESULTS_PATH,
    start_code_index = 1,
    end_code_index = nothing,
    max_codes = nothing,
    n_samples = 1_000,
    repeats = 1,
    target_workers = 8,
    sigmas = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    sigmas_already_normalized = false,
    decoder = "lsd",
    schedule = "serial",
    decoding_style = "received_vector",
    search_radius = 1.0,
    local_search = true,
    local_search_lll = false,
    sphere_decoding = false,
    full_basis = false,
)

function run_with_script_settings(settings = RUN_SETTINGS)
    sigmas = normalize_sigmas(
        settings.sigmas;
        already_normalized = settings.sigmas_already_normalized,
    )

    run_generated_codes_decoding(
        source_root = settings.source_root,
        reduced_codes_dir = settings.reduced_codes_dir,
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
    run_with_script_settings()
end
