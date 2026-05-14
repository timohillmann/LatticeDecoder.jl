using Distributed

include(joinpath(@__DIR__, "data_collection_utils.jl"))

ensure_data_collection_workers!()

using NPZ
using SparseArrays: sparse, SparseMatrixCSC

@everywhere using LinearAlgebra
@everywhere using LatticeDecoder

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_PATH = "results/bivariate_bicycle/bivariate_bicycle_css_decoding.csv"

const CODE_FAMILY_DIRS = Dict(
    "bivariate_bicycle" => joinpath(REPO_ROOT, "generator_matrices", "bivariate_bicycle"),
    "simplex_codes" => joinpath(REPO_ROOT, "generator_matrices", "simplex_codes"),
)

function balance_row_weights!(mat::SparseMatrixCSC)
    for i in 1:size(mat, 1)
        row = mat[i, :]
        if length(row.nzind) == 1
            nz_idx = row.nzind[1]
            for j in 1:size(mat, 1)
                j == i && continue
                if !iszero(mat[j, nz_idx])
                    row2 = mat[j, :]
                    mat[i, :] .= row2 .- row
                    break
                end
            end
        end
    end

    return mat
end

function code_family_from_path(path::AbstractString)
    normalized_path = normpath(path)
    for (family, root) in CODE_FAMILY_DIRS
        if startswith(normalized_path, normpath(root))
            return family
        end
    end

    return "unknown"
end

function css_code_path(
    code_name_or_path::AbstractString;
    code_family::Union{Nothing,AbstractString} = nothing,
    reduced_basis::Bool = true,
)
    if isfile(code_name_or_path)
        return code_name_or_path
    end

    suffix = reduced_basis ? "_expanded.npz" : "_overcomplete.npz"
    filename = endswith(code_name_or_path, ".npz") ? code_name_or_path : "$(code_name_or_path)$(suffix)"
    families = code_family === nothing ? collect(keys(CODE_FAMILY_DIRS)) : [String(code_family)]

    for family in families
        haskey(CODE_FAMILY_DIRS, family) || throw(ArgumentError("unknown code family $(family)"))
        path = joinpath(CODE_FAMILY_DIRS[family], "expanded", filename)
        isfile(path) && return path
    end

    searched = [joinpath(CODE_FAMILY_DIRS[family], "expanded", filename) for family in families]
    throw(ArgumentError("could not find expanded CSS code $(code_name_or_path); searched $(searched)"))
end

function css_code_prime(code_name::AbstractString)
    match_result = match(r"_p(\d+)", code_name)
    match_result === nothing && return 2
    return parse(Int, match_result.captures[1])
end

function css_code_distance(code_name::AbstractString)
    match_result = match(r"^\d+_\d+_(\d+)_p\d+$", code_name)
    match_result === nothing && throw(ArgumentError("could not parse distance from code name $(code_name); expected n_k_d_p format"))
    return parse(Int, match_result.captures[1])
end

function local_search_order_for_code(code_name::AbstractString)
    distance = css_code_distance(code_name)
    return [2; fill(1, distance - 1)]
end

function canonical_code_name(path::AbstractString)
    return replace(basename(path), r"_(expanded|overcomplete)\.npz$" => "")
end

function load_code(
    code_name::String,
    balance_weights::Bool = false,
    reduced_basis::Bool = true;
    code_family::Union{Nothing,AbstractString} = nothing,
)
    code_path = css_code_path(code_name; code_family, reduced_basis)
    data = npzread(code_path)
    canonical_name = canonical_code_name(code_path)
    p = css_code_prime(canonical_name)

    if balance_weights
        H = data["Mqq"]
        H_CSC = sparse(H)
        balance_row_weights!(H_CSC)
        Mqq = Matrix{Float64}(H_CSC ./ sqrt(p))

        H = data["Mpp"]
        H_CSC = sparse(H)
        balance_row_weights!(H_CSC)
        Mpp = Matrix{Float64}(H_CSC ./ sqrt(p))

        return Mqq, Mpp
    end

    Mqq = Matrix{Float64}(data["Mqq"]) ./ sqrt(p)
    Mpp = Matrix{Float64}(data["Mpp"]) ./ sqrt(p)

    return Mqq, Mpp
end

function bivariate_bicycle_css_problem(
    code_name_or_path::AbstractString,
    basis::AbstractString;
    code_family::Union{Nothing,AbstractString} = nothing,
    balance_weights::Bool = false,
    reduced_basis::Bool = true,
)
    code_path = css_code_path(code_name_or_path; code_family, reduced_basis)
    code_name = canonical_code_name(code_path)
    p = css_code_prime(code_name)
    distance = css_code_distance(code_name)
    family = code_family === nothing ? code_family_from_path(code_path) : String(code_family)
    Mqq, Mpp = load_code(
        String(code_name_or_path),
        balance_weights,
        reduced_basis;
        code_family = family,
    )

    if basis == "X"
        return (
            code_name = code_name,
            code_family = family,
            code_path = code_path,
            p = p,
            distance = distance,
            H = Mqq,
            G = inv(Mpp),
            logical_check = inv(Mqq),
        )
    elseif basis == "Z"
        return (
            code_name = code_name,
            code_family = family,
            code_path = code_path,
            p = p,
            distance = distance,
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

function experiment_metadata(params::Dict, σ::Float64, nbits::Int, code_name::String, code_family::String, p::Int, distance::Int)
    meta = Dict{Symbol,Any}()
    for (key, value) in params
        meta[key] = value
    end

    meta[:sigma] = σ
    meta[:nbits] = nbits
    meta[:code_name] = code_name
    meta[:code_family] = code_family
    meta[:p] = p
    meta[:distance] = distance
    meta[:css_decoding] = true

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
                problem = bivariate_bicycle_css_problem(
                    code_name,
                    params[:basis];
                    code_family = get(params, :code_family, nothing),
                    balance_weights = params[:balance_weights],
                    reduced_basis = params[:reduced_basis],
                )
                H = problem.H
                G = problem.G
                logical_check = problem.logical_check

                params[:iterations] = get(params, :iterations, size(H, 2))
                local_search_order = params[:local_search] ? local_search_order_for_code(problem.code_name) : [0]

                for σ in params[:sigmas]
                    meta = experiment_metadata(params, σ, size(H, 2), problem.code_name, problem.code_family, problem.p, problem.distance)
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
        "30_8_4_p2",
        "62_10_6_p2",
        "126_12_10_p2",
        # "254_14_16_p2",
        # "510_16_24_p2",
    ]

    param_ranges = Dict(
        :search_radius => [1.0],
        :decoder => ["lsd"],
        :schedule => ["serial", "parallel"],
        :sigmas => [collect(0.3:0.05:0.8) ./ sqrt(2π)],
        :basis => ["X", "Z"],
        :balance_weights => [false, true],
        :reduced_basis => [true],
        :local_search => [false, true],
        :local_search_lll => [false],
        :sphere_decoding => [false, true],
        :full_basis => [false],
    )

    n_samples = 1_000
    repeats = 10

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
