using Distributed
using LinearAlgebra

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "data_collection_utils.jl"))

ensure_data_collection_workers!()

include(joinpath(REPO_ROOT, "experiments", "decode_qldlc_codes.jl"))

@everywhere using LinearAlgebra
@everywhere using LatticeDecoder

function run_samples(;
    H::AbstractMatrix{Float64},
    H_reduced::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical_check::AbstractMatrix{Float64},
    σ::Float64,
    n_samples::Int,
    schedule::String = "serial",
    iterations::Int = size(H_reduced, 2),
    decoder::Union{String,Int64} = "lsd",
    search_radius::Float64 = 1.0,
    lsd_beta::Float64 = LatticeDecoder.LSD_DEFAULT_BETA,
    lsd_w_min::Float64 = LatticeDecoder.LSD_W_MIN,
    local_search::Bool = false,
    local_search_order::Vector{Int64} = [1],
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
        lsd_beta = lsd_beta,
        lsd_w_min = lsd_w_min,
    )

    return @distributed (+) for _ in 1:n_samples
        error_vector = sample_error(σ, size(H_reduced, 2))
        received = copy(error_vector)


        bp_estimate = run_decoder!(ldlc_decoder, received)
        decoded_integer_correction = hard_decision(bp_estimate, H_reduced)

        if local_search
            λ = abs.(H_reduced * bp_estimate) .% 1.0
            local_search!(received, λ, decoded_integer_correction, lsd)
        end

        correction = received - G * decoded_integer_correction
        residual = error_vector - correction

        is_not_logical_error(logical_check, residual) ? 0 : 1
    end
end

function qldlc_problem(code_name_or_path::AbstractString)
    code_path = qldlc_generated_code_path(code_name_or_path)
    code = load_qldlc_code(code_path)
    code_name = splitext(basename(code_path))[1]

    M_classical = Float64.(Matrix(code["classical_generator"]));
    M_reduced = Float64.(Matrix(code["qubit_generator_lll"]));
    logical_rows = Float64.(vcat(code["XL"]', code["ZL"]'));

    M = vcat(M_classical, 2 .*  logical_rows);

    J = symplectic_form(size(M_reduced, 2) ÷ 2)
    H = -M * J
    H_reduced = -M_reduced * J
    G = J * inv(M_reduced)
    logical_check = inv(M_reduced)

    return (
        code_name = code_name,
        H = Matrix(H),
        H_reduced = Matrix(H_reduced),
        G = Matrix(G),
        logical_check = Matrix(logical_check),
    )
end

function experiment_metadata(params::Dict, σ::Float64, nbits::Int, code_name::String)
    meta = Dict{Symbol,Any}(
        :decoder => params[:decoder],
        :schedule => params[:schedule],
        :search_radius => params[:search_radius],
        :lsd_beta => get(params, :lsd_beta, LatticeDecoder.LSD_DEFAULT_BETA),
        :lsd_w_min => get(params, :lsd_w_min, LatticeDecoder.LSD_W_MIN),
        :iterations => params[:iterations],
        :sigma => σ,
        :nbits => nbits,
        :code_name => code_name,
    )

    
    meta[:local_search] = params[:local_search]
    meta[:local_search_order] = params[:local_search_order]
    meta[:local_search_lll] = params[:local_search_lll]
    meta[:sphere_decoding] = params[:sphere_decoding]
    meta[:full_basis] = params[:full_basis]

    return meta
end

function run_qldlc_experiment!(
    path::AbstractString;
    code_names::AbstractVector{<:AbstractString},
    param_ranges::Dict,
    n_samples::Int,
    repeats::Int,
    max_errors::Union{Nothing,Int} = DEFAULT_MAX_ERRORS,
)
    accumulated_errors = accumulated_errors_by_strong_id(path)

    for repeat_index in 1:repeats
        for code_name in code_names
            problem = qldlc_problem(code_name)

            for params in parameter_grid(param_ranges)
                params[:iterations] = get(params, :iterations, size(problem.H_reduced, 2))


                for σ in params[:sigmas]
                    meta = experiment_metadata(
                        params,
                        σ,
                        size(problem.H_reduced, 2),
                        problem.code_name,
                    )
                    strong_id = LatticeDecoder.get_strong_id_from_json(meta)

                    if max_errors !== nothing && get(accumulated_errors, strong_id, 0) >= max_errors
                        continue
                    end

                    errors = run_samples(;
                        H = problem.H,
                        H_reduced = problem.H_reduced,
                        G = problem.G,
                        logical_check = problem.logical_check,
                        σ,
                        n_samples,
                        schedule = params[:schedule],
                        iterations = params[:iterations],
                        decoder = params[:decoder],
                        search_radius = params[:search_radius],
                        lsd_beta = get(params, :lsd_beta, LatticeDecoder.LSD_DEFAULT_BETA),
                        lsd_w_min = get(params, :lsd_w_min, LatticeDecoder.LSD_W_MIN),
                        local_search = params[:local_search],
                        local_search_order = params[:local_search_order],
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
            end

            flush(stdout)
        end
    end
end

function main()
    path = "results/qldlc/generated_qldlc_overcomplete.csv"

code_names = [
    "reduced_ldlc_gkp_n_13_1",
    "reduced_ldlc_gkp_n_13_2",
    "reduced_ldlc_gkp_n_13_3",
    "reduced_ldlc_gkp_n_13_4",
    "reduced_ldlc_gkp_n_13_5",
    "reduced_ldlc_gkp_n_13_6",
    "reduced_ldlc_gkp_n_13_7",
    "reduced_ldlc_gkp_n_13_8",
    "reduced_ldlc_gkp_n_13_9",
    "reduced_ldlc_gkp_n_13_10",
    "reduced_ldlc_gkp_n_13_11",
    "reduced_ldlc_gkp_n_13_12",
    "reduced_ldlc_gkp_n_13_13",
    "reduced_ldlc_gkp_n_13_14",
    "reduced_ldlc_gkp_n_13_15",
    "reduced_ldlc_gkp_n_13_16",
    "reduced_ldlc_gkp_n_13_17",
    "reduced_ldlc_gkp_n_13_18",
    "reduced_ldlc_gkp_n_13_19",
    "reduced_ldlc_gkp_n_13_20",
    "reduced_ldlc_gkp_n_13_21",
    "reduced_ldlc_gkp_n_13_22",
    "reduced_ldlc_gkp_n_13_23",
    "reduced_ldlc_gkp_n_13_24",
    "reduced_ldlc_gkp_n_13_25",
    "reduced_ldlc_gkp_n_13_26",
    "reduced_ldlc_gkp_n_13_27",
    "reduced_ldlc_gkp_n_13_28",
    "reduced_ldlc_gkp_n_13_29",
    "reduced_ldlc_gkp_n_13_30",
    "reduced_ldlc_gkp_n_14_1",
    "reduced_ldlc_gkp_n_14_2",
    "reduced_ldlc_gkp_n_14_3",
    "reduced_ldlc_gkp_n_14_4",
    "reduced_ldlc_gkp_n_14_5",
    "reduced_ldlc_gkp_n_14_6",
    "reduced_ldlc_gkp_n_14_7",
    "reduced_ldlc_gkp_n_14_8",
    "reduced_ldlc_gkp_n_14_9",
    "reduced_ldlc_gkp_n_14_10",
    "reduced_ldlc_gkp_n_14_11",
    "reduced_ldlc_gkp_n_14_12",
    "reduced_ldlc_gkp_n_14_13",
    "reduced_ldlc_gkp_n_14_14",
    "reduced_ldlc_gkp_n_14_15",
    "reduced_ldlc_gkp_n_14_16",
    "reduced_ldlc_gkp_n_14_17",
    "reduced_ldlc_gkp_n_14_18",
    "reduced_ldlc_gkp_n_14_19",
    "reduced_ldlc_gkp_n_14_20",
    "reduced_ldlc_gkp_n_14_21",
    "reduced_ldlc_gkp_n_14_22",
    "reduced_ldlc_gkp_n_14_23",
    "reduced_ldlc_gkp_n_14_24",
    "reduced_ldlc_gkp_n_14_25",
    "reduced_ldlc_gkp_n_14_26",
    "reduced_ldlc_gkp_n_14_27",
    "reduced_ldlc_gkp_n_14_28",
    "reduced_ldlc_gkp_n_14_29",
    "reduced_ldlc_gkp_n_14_30",
    "reduced_ldlc_gkp_n_15_1",
    "reduced_ldlc_gkp_n_15_2",
    "reduced_ldlc_gkp_n_15_3",
    "reduced_ldlc_gkp_n_15_4",
    "reduced_ldlc_gkp_n_15_5",
    "reduced_ldlc_gkp_n_15_6",
    "reduced_ldlc_gkp_n_15_7",
    "reduced_ldlc_gkp_n_15_8",
    "reduced_ldlc_gkp_n_15_9",
    "reduced_ldlc_gkp_n_15_10",
    "reduced_ldlc_gkp_n_15_11",
    "reduced_ldlc_gkp_n_15_12",
    "reduced_ldlc_gkp_n_15_13",
    "reduced_ldlc_gkp_n_15_14",
    "reduced_ldlc_gkp_n_15_15",
    "reduced_ldlc_gkp_n_15_16",
    "reduced_ldlc_gkp_n_15_17",
    "reduced_ldlc_gkp_n_15_18",
    "reduced_ldlc_gkp_n_15_19",
    "reduced_ldlc_gkp_n_15_20",
    "reduced_ldlc_gkp_n_15_21",
    "reduced_ldlc_gkp_n_15_22",
    "reduced_ldlc_gkp_n_15_23",
    "reduced_ldlc_gkp_n_15_24",
    "reduced_ldlc_gkp_n_15_25",
    "reduced_ldlc_gkp_n_15_26",
    "reduced_ldlc_gkp_n_15_27",
    "reduced_ldlc_gkp_n_15_28",
    "reduced_ldlc_gkp_n_15_29",
    "reduced_ldlc_gkp_n_15_30",
    "reduced_ldlc_gkp_n_16_1",
    "reduced_ldlc_gkp_n_16_2",
    "reduced_ldlc_gkp_n_16_3",
    "reduced_ldlc_gkp_n_16_5",
    "reduced_ldlc_gkp_n_16_6",
    "reduced_ldlc_gkp_n_16_7",
    "reduced_ldlc_gkp_n_16_8",
    "reduced_ldlc_gkp_n_16_9",
    "reduced_ldlc_gkp_n_16_10",
    "reduced_ldlc_gkp_n_16_12",
    "reduced_ldlc_gkp_n_16_13",
    "reduced_ldlc_gkp_n_16_14",
    "reduced_ldlc_gkp_n_16_15",
    "reduced_ldlc_gkp_n_16_16",
    "reduced_ldlc_gkp_n_16_17",
    "reduced_ldlc_gkp_n_16_18",
    "reduced_ldlc_gkp_n_16_19",
    "reduced_ldlc_gkp_n_16_20",
    "reduced_ldlc_gkp_n_16_21",
    "reduced_ldlc_gkp_n_16_22",
    "reduced_ldlc_gkp_n_16_23",
    "reduced_ldlc_gkp_n_16_24",
    "reduced_ldlc_gkp_n_16_25",
    "reduced_ldlc_gkp_n_16_26",
    "reduced_ldlc_gkp_n_16_27",
    "reduced_ldlc_gkp_n_16_28",
    "reduced_ldlc_gkp_n_16_29",
    "reduced_ldlc_gkp_n_16_30",
    "reduced_ldlc_gkp_n_17_1",
    "reduced_ldlc_gkp_n_17_5",
    "reduced_ldlc_gkp_n_17_17",
    "reduced_ldlc_gkp_n_17_18",
    "reduced_ldlc_gkp_n_17_23",
    "reduced_ldlc_gkp_n_17_24",
    "reduced_ldlc_gkp_n_17_25",
    "reduced_ldlc_gkp_n_17_26",
    "reduced_ldlc_gkp_n_17_30",
]

    param_ranges = Dict(
        :search_radius => [1.0],
        :decoder => ["lsd"],
        :schedule => ["serial"],
        :sigmas => [[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6] ./ sqrt(2π)],
        :local_search => [true],
        :local_search_order => [[1, 1, 1, 1, 1, 1, 1, 1]],
        :local_search_lll => [false],
        :sphere_decoding => [false, true],
        :full_basis => [false],
    )

    n_samples = 10_000
    repeats = 10

    run_qldlc_experiment!(
        path;
        code_names,
        param_ranges,
        n_samples,
        repeats,
        max_errors = 200,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
