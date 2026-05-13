using Distributed

include(joinpath(@__DIR__, "data_collection_utils.jl"))

ensure_data_collection_workers!()

@everywhere using LinearAlgebra
@everywhere using LatticeDecoder

const OUTPUT_PATH = "results/rep_code/rep_code_decoding.csv"

function build_repetition_check_matrix(n::Int, balance_mode::AbstractString)
    H = reduced_repetition_sector(n)

    if balance_mode == "unbalanced"
        return H
    elseif balance_mode == "balance_first"
        H[1, 1:2] = [1.0, -1.0] ./ sqrt(2)
    elseif balance_mode == "balance_last"
        H[1, 1] = 1.0 / sqrt(2)
        H[1, 2] = 0.0
        H[1, end] = 1.0 / sqrt(2)
    else
        error("Unknown balance_mode: $(balance_mode)")
    end

    return H
end

function reduced_repetition_sector(n::Int)
    try
        code = GKP_Rep_Code(n, false, true)
        return Matrix{Float64}(code.code[(n + 1):end, (n + 1):end])
    catch err
        err isa UndefVarError && err.var === :stack_gkp_generator || rethrow()
        return direct_reduced_repetition_sector(n)
    end
end

function direct_reduced_repetition_sector(n::Int)
    H = zeros(Float64, n, n)
    for row in 1:(n - 1)
        H[row, row] = 1.0 / sqrt(2)
        H[row, row + 1] = 1.0 / sqrt(2)
    end
    H[n, n] = sqrt(2)
    return H
end

@everywhere function run_repetition_samples(;
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
    meta[:reduced_decoding] = true
    meta[:bit_flip] = false

    delete!(meta, :sigmas)

    return meta
end

function run_repetition_experiment!(
    path::AbstractString;
    param_ranges::Dict,
    n_samples::Int,
    repeats::Int,
    max_errors::Union{Nothing,Int} = DEFAULT_MAX_ERRORS,
)
    accumulated_errors = accumulated_errors_by_strong_id(path)

    for _ in 1:repeats
        for params in parameter_grid(param_ranges)
            H = build_repetition_check_matrix(params[:n], params[:balance_mode])
            G = round.(sqrt(2) * inv(H)) ./ sqrt(2)
            logical_check = inv(H)

            params[:iterations] = get(params, :iterations, size(H, 2))
            local_search_order = params[:local_search] ? [2; fill(1, params[:n]÷2 - 1)] : [0]

            for σ in params[:sigmas]
                meta = experiment_metadata(params, σ, size(H, 2))
                meta[:local_search_order] = local_search_order
                strong_id = LatticeDecoder.get_strong_id_from_json(meta)

                if max_errors !== nothing && get(accumulated_errors, strong_id, 0) >= max_errors
                    continue
                end

                errors = run_repetition_samples(;
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
        :n => [3, 5],
        :sigmas => [collect(0.5:0.05:1.5) ./ sqrt(2π)],
        :balance_mode => ["unbalanced", "balance_first", "balance_last"],
        :local_search => [false, true],
        :local_search_lll => [false],
        :sphere_decoding => [false, true],
        :full_basis => [false],
    )

    n_samples = 10_000
    repeats = 1

    run_repetition_experiment!(
        OUTPUT_PATH;
        param_ranges,
        n_samples,
        repeats,
        max_errors = DEFAULT_MAX_ERRORS,
    )
end

main()
