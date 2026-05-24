using Distributed
using LinearAlgebra
using LatticeDecoder

include(joinpath(@__DIR__, "data_collection_utils.jl"))

ensure_data_collection_workers!()

@everywhere using LinearAlgebra
@everywhere using LatticeDecoder

const OUTPUT_PATH = "results/classical_ldlc/classical_ldlc_decoding.csv"
const CLASSICAL_LDLC_DIR = joinpath(@__DIR__, "..", "generator_matrices", "classical_ldlc")

function parse_classical_ldlc_code_name(code_name::AbstractString)
    match_result = match(r"^ldlc_n(\d+)_d(\d+)$", code_name)
    match_result === nothing && error("Invalid classical LDLC code name: $(code_name)")
    return (
        n = parse(Int, match_result.captures[1]),
        d = parse(Int, match_result.captures[2]),
    )
end

function available_classical_ldlc_code_names()
    isdir(CLASSICAL_LDLC_DIR) || error("Classical LDLC directory not found: $(CLASSICAL_LDLC_DIR)")

    code_names = String[]
    for file_name in readdir(CLASSICAL_LDLC_DIR)
        match_result = match(r"^(ldlc_n\d+_d\d+)_H\.npy$", file_name)
        match_result === nothing && continue

        code_name = match_result.captures[1]
        g_path = joinpath(CLASSICAL_LDLC_DIR, "$(code_name)_G.npy")
        isfile(g_path) && push!(code_names, code_name)
    end

    return sort(unique(code_names); by = name -> begin
        code = parse_classical_ldlc_code_name(name)
        (code.n, code.d)
    end)
end

function load_classical_ldlc_problem(code_name::AbstractString)
    code = parse_classical_ldlc_code_name(code_name)
    H, G, loaded_code_name = LatticeDecoder.load_classical_ldlc(Dict("n" => code.n, "d" => code.d))
    return (
        code_name = loaded_code_name,
        n = code.n,
        d = code.d,
        H = Matrix{Float64}(H),
        G = Matrix{Float64}(G),
    )
end

@everywhere function run_classical_ldlc_samples(;
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    σ::Float64,
    n_samples::Int,
    schedule::String = "parallel",
    iterations::Int = 25,
    decoder::Union{String,Int64} = "lsd",
    search_radius::Float64 = 1.5,
    local_search::Bool = false,
    local_search_width::Int = 10,
    local_search_order::Vector{Int64} = [2, 1, 1],
    local_search_lll::Bool = false,
    sphere_decoding::Bool = true,
    full_basis::Bool = false,
)
    n = size(H, 1)
    effective_order = local_search ? local_search_order : [0]
    effective_width = local_search ? local_search_width : 1

    lsd = LocalSearch(
        effective_width,
        G,
        effective_order,
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

    return data_collection_sample_sum(n_samples) do _
        b = zeros(Int64, n)
        random_bitstring!(b, n)

        received = encode(b, G)
        received .+= sample_error(σ, n)

        bp_estimate = run_decoder!(ldlc_decoder, received)
        decoded_word = hard_decision(bp_estimate, H)

        if local_search
            λ = abs.(H * bp_estimate) .% 1.0
            local_search!(received, λ, decoded_word, lsd)
        end

        count_symbol_errors(decoded_word, b)
    end
end

function experiment_metadata(params::Dict, σ::Float64, problem)
    meta = Dict{Symbol,Any}()
    for (key, value) in params
        meta[key] = value
    end

    meta[:sigma] = σ
    meta[:nbits] = problem.n
    meta[:n] = problem.n
    meta[:d] = problem.d
    meta[:code_name] = problem.code_name
    meta[:code_family] = "classical_ldlc"
    meta[:code_source] = "generator_matrices/classical_ldlc"
    meta[:shots_are_symbols] = true
    meta[:effective_local_search] = params[:local_search]
    meta[:effective_local_search_width] = params[:local_search] ? params[:local_search_width] : 1
    meta[:effective_local_search_order] = params[:local_search] ? params[:local_search_order] : [0]

    delete!(meta, :sigmas)

    return meta
end

function run_classical_ldlc_experiment!(
    path::AbstractString;
    code_names::AbstractVector{<:AbstractString},
    param_ranges::Dict,
    n_samples::Int,
    repeats::Int,
    max_errors::Union{Nothing,Int} = 500,
)
    accumulated_errors = accumulated_errors_by_strong_id(path)

    for _ in 1:repeats
        for code_name in code_names
            problem = load_classical_ldlc_problem(code_name)

            for params in parameter_grid(param_ranges)
                params[:iterations] = get(params, :iterations, 25)

                for σ in params[:sigmas]
                    meta = experiment_metadata(params, σ, problem)
                    strong_id = LatticeDecoder.get_strong_id_from_json(meta)

                    if max_errors !== nothing && get(accumulated_errors, strong_id, 0) >= max_errors
                        continue
                    end

                    errors = run_classical_ldlc_samples(;
                        H = problem.H,
                        G = problem.G,
                        σ,
                        n_samples,
                        schedule = params[:schedule],
                        iterations = params[:iterations],
                        decoder = params[:decoder],
                        search_radius = params[:search_radius],
                        local_search = params[:local_search],
                        local_search_width = params[:local_search_width],
                        local_search_order = params[:local_search_order],
                        local_search_lll = params[:local_search_lll],
                        sphere_decoding = params[:sphere_decoding],
                        full_basis = params[:full_basis],
                    )

                    symbol_trials = n_samples * problem.n
                    add_data!(
                        String(path);
                        shots = symbol_trials,
                        errors = errors,
                        decoder = string(params[:decoder]),
                        json_metadata = meta,
                        custom_counts = Dict("codeword_samples" => n_samples),
                    )
                    accumulated_errors[strong_id] = get(accumulated_errors, strong_id, 0) + errors
                end

                flush(stdout)
            end
        end
    end
end

function main()
    σ_capacity = lattice_capacity_std()
    code_names = available_classical_ldlc_code_names()

    param_ranges = Dict(
        :search_radius => [1.5],
        :decoder => ["lsd", "nearest"],
        :schedule => ["parallel", "serial"],
        :sigmas => [collect(range(σ_capacity, 0.8 * σ_capacity; length = 11))],
        :local_search => [false, true],
        :local_search_width => [10],
        :local_search_order => [[2, 1, 1]],
        :local_search_lll => [false],
        :sphere_decoding => [true, false],
        :full_basis => [false],
    )

    n_samples = 1_000
    repeats = 10

    run_classical_ldlc_experiment!(
        OUTPUT_PATH;
        code_names,
        param_ranges,
        n_samples,
        repeats,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
