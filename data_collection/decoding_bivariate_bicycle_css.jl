using Distributed

include(joinpath(@__DIR__, "data_collection_utils.jl"))

ensure_data_collection_workers!()

using NPZ

@everywhere using LinearAlgebra
@everywhere using LatticeDecoder

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const BB_EXPANDED_DIR = joinpath(REPO_ROOT, "generator_matrices", "bivariate_bicycle", "expanded")
const OUTPUT_PATH = "results/bivariate_bicycle/bivariate_bicycle_css_decoding.csv"

function bivariate_bicycle_code_path(code_name_or_path::AbstractString)
    if isfile(code_name_or_path)
        return code_name_or_path
    end

    filename = endswith(code_name_or_path, ".npz") ? code_name_or_path : "$(code_name_or_path)_expanded.npz"
    path = joinpath(BB_EXPANDED_DIR, filename)
    isfile(path) || throw(ArgumentError("could not find bivariate bicycle expanded code at $(path)"))
    return path
end

function bivariate_bicycle_prime(code_name::AbstractString)
    match_result = match(r"_p(\d+)", code_name)
    match_result === nothing && return 2
    return parse(Int, match_result.captures[1])
end

function bivariate_bicycle_css_problem(code_name_or_path::AbstractString, basis::AbstractString)
    code_path = bivariate_bicycle_code_path(code_name_or_path)
    data = npzread(code_path)
    code_name = replace(basename(code_path), r"_expanded\.npz$" => "")
    p = bivariate_bicycle_prime(code_name)

    Mqq = Matrix{Float64}(data["hx"]) ./ sqrt(p)
    Mpp = Matrix{Float64}(data["hz"]) ./ sqrt(p)

    if basis == "X"
        return (
            code_name = code_name,
            code_path = code_path,
            p = p,
            H = Mqq,
            G = inv(Mpp),
            logical_check = inv(Mqq),
        )
    elseif basis == "Z"
        return (
            code_name = code_name,
            code_path = code_path,
            p = p,
            H = Mpp,
            G = -inv(Mqq),
            logical_check = inv(Mpp),
        )
    else
        error("Unknown CSS basis: $(basis)")
    end
end

@everywhere function run_bivariate_bicycle_css_samples(;
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical_check::AbstractMatrix{Float64},
    σ::Float64,
    n_samples::Int,
    schedule::String = "serial",
    iterations::Int = size(H, 2),
    decoder::Union{String,Int64} = "lsd",
    search_radius::Float64 = 1.0,
    local_search::Bool = false,
    local_search_order::Vector{Int64} = [0],
    local_search_lll::Bool = false,
    sphere_decoding::Bool = false,
    full_basis::Bool = false,
)
    lsd = LocalSearch(
        length(local_search_order),
        G,
        local_search_order,
        local_search_lll,
        sphere_decoding,
        full_basis,
    )

    tanner_graph = initialize_tanner_graph(H)
    ldlc_decoder = LDLCDecoder(
        tanner_graph;
        schedule = schedule,
        algorithm = decoder,
        sigma = σ,
        max_iterations = iterations,
        search_interval = search_radius,
    )

    return @distributed (+) for _ in 1:n_samples
        error_vector = sample_error(σ, size(H, 2))
        received = copy(error_vector)

        bp_estimate = run_decoder!(ldlc_decoder, received)
        decoded_integer_correction = hard_decision(bp_estimate, H)

        if local_search
            λ = abs.(H * bp_estimate) .% 1.0
            local_search!(received, λ, decoded_integer_correction, lsd)
        end

        correction = received - G * decoded_integer_correction
        residual = error_vector - correction

        is_not_logical_error(logical_check, residual) ? 0 : 1
    end
end

function experiment_metadata(params::Dict, σ::Float64, nbits::Int, code_name::String, p::Int)
    meta = Dict{Symbol,Any}()
    for (key, value) in params
        meta[key] = value
    end

    meta[:sigma] = σ
    meta[:nbits] = nbits
    meta[:code_name] = code_name
    meta[:p] = p
    meta[:css_decoding] = true
    meta[:expanded] = true

    delete!(meta, :sigmas)

    return meta
end

function run_bivariate_bicycle_css_experiment!(
    path::AbstractString;
    code_names::AbstractVector{<:AbstractString},
    param_ranges::Dict,
    n_samples::Int,
    repeats::Int,
    max_errors::Union{Nothing,Int} = DEFAULT_MAX_ERRORS,
)
    accumulated_errors = accumulated_errors_by_strong_id(path)

    for _ in 1:repeats
        for code_name in code_names
            for params in parameter_grid(param_ranges)
                problem = bivariate_bicycle_css_problem(code_name, params[:basis])
                H = problem.H
                G = problem.G
                logical_check = problem.logical_check

                params[:iterations] = get(params, :iterations, size(H, 2))
                local_search_order = params[:local_search] ? params[:local_search_order] : [0]

                for σ in params[:sigmas]
                    meta = experiment_metadata(params, σ, size(H, 2), problem.code_name, problem.p)
                    meta[:local_search_order] = local_search_order
                    strong_id = LatticeDecoder.get_strong_id_from_json(meta)

                    if max_errors !== nothing && get(accumulated_errors, strong_id, 0) >= max_errors
                        continue
                    end

                    errors = run_bivariate_bicycle_css_samples(;
                        H,
                        G,
                        logical_check,
                        σ,
                        n_samples,
                        schedule = params[:schedule],
                        iterations = params[:iterations],
                        decoder = params[:decoder],
                        search_radius = params[:search_radius],
                        local_search = params[:local_search],
                        local_search_order,
                        local_search_lll = params[:local_search_lll],
                        sphere_decoding = params[:sphere_decoding],
                        full_basis = params[:full_basis],
                    )

                    add_data!(
                        String(path);
                        shots = n_samples,
                        errors = errors,
                        decoder = string(params[:decoder]),
                        json_metadata = meta,
                    )
                    accumulated_errors[strong_id] = get(accumulated_errors, strong_id, 0) + errors
                end

                flush(stdout)
            end
        end
    end
end

function main()
    code_names = [
        "30_4_5_p2",
        "48_4_7_p2",
        "78_4_9_p2",
    ]

    param_ranges = Dict(
        :search_radius => [1.0],
        :decoder => ["lsd"],
        :schedule => ["serial", "parallel"],
        :sigmas => [collect(0.3:0.05:0.8) ./ sqrt(2π)],
        :basis => ["X"],
        :local_search => [false, true],
        :local_search_order => [[1]],
        :local_search_lll => [false],
        :sphere_decoding => [false, true],
        :full_basis => [false],
    )

    n_samples = 10_000
    repeats = 1

    run_bivariate_bicycle_css_experiment!(
        OUTPUT_PATH;
        code_names,
        param_ranges,
        n_samples,
        repeats,
        max_errors = DEFAULT_MAX_ERRORS,
    )
end

main()
