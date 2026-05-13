using Distributed

include(joinpath(@__DIR__, "data_collection_utils.jl"))

ensure_data_collection_workers!()

@everywhere using LinearAlgebra
@everywhere using LatticeDecoder

const OUTPUT_PATH = "results/surface_code/surface_code_noncss_decoding.csv"

function noncss_surface_code_problem(d::Int)
    code = GKP_Surface_Code(d, false)
    M = Matrix{Float64}(code.code)
    J = code.J
    inv_M = inv(M)
    return (
        H = -M * J,
        G = J * inv_M,
        logical_check = inv_M  # For non-CSS surface code, the logical check matrix can be taken as the inverse of the full code matrix M
    )
end

@everywhere function run_surface_code_noncss_samples(;
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

function experiment_metadata(params::Dict, σ::Float64, nbits::Int)
    meta = Dict{Symbol,Any}()
    for (key, value) in params
        meta[key] = value
    end

    meta[:sigma] = σ
    meta[:nbits] = nbits
    meta[:css_decoding] = false
    meta[:balance_hamming_weight] = false
    meta[:bit_flip] = false

    delete!(meta, :sigmas)

    return meta
end

function run_surface_code_noncss_experiment!(
    path::AbstractString;
    param_ranges::Dict,
    n_samples::Int,
    repeats::Int,
    max_errors::Union{Nothing,Int} = DEFAULT_MAX_ERRORS,
)
    accumulated_errors = accumulated_errors_by_strong_id(path)

    for _ in 1:repeats
        for params in parameter_grid(param_ranges)
            problem = noncss_surface_code_problem(params[:d])
            H = problem.H
            G = problem.G
            logical_check = problem.logical_check

            params[:iterations] = get(params, :iterations, size(H, 2))
            local_search_order = params[:local_search] ? [2; fill(1, params[:d] - 1)] : [0]

            for σ in params[:sigmas]
                meta = experiment_metadata(params, σ, size(H, 2))
                meta[:local_search_order] = local_search_order
                strong_id = LatticeDecoder.get_strong_id_from_json(meta)

                if max_errors !== nothing && get(accumulated_errors, strong_id, 0) >= max_errors
                    continue
                end

                errors = run_surface_code_noncss_samples(;
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

function main()
    param_ranges = Dict(
        :search_radius => [1.0],
        :decoder => ["lsd", "nearest"],
        :schedule => ["serial", "parallel"],
        :d => [3, 5],
        :sigmas => [[1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5] ./ sqrt(2π)],
        :local_search => [false, true],
        :local_search_lll => [false],
        :sphere_decoding => [false, true],
        :full_basis => [false],
    )

    n_samples = 10_000
    repeats = 1

    run_surface_code_noncss_experiment!(
        OUTPUT_PATH;
        param_ranges,
        n_samples,
        repeats,
        max_errors = DEFAULT_MAX_ERRORS,
    )
end

main()
