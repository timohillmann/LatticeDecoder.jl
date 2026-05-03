struct QECProblem
    H::Matrix{Float64}
    G::Matrix{Float64}
    logical_check::Matrix{Float64}
    order::Vector{Int64}
end

function QECProblem(H::AbstractMatrix{Float64}, G::AbstractMatrix{Float64}, order::AbstractVector{Int64})
    return QECProblem(Matrix(H), Matrix(G), inv(H), collect(order))
end

function is_not_logical_error(logical_check::AbstractMatrix{Float64}, residual::AbstractVector{Float64}, eps::Float64=1e-5)
    log_check = logical_check' * residual
    return all(abs(x - round(x)) < eps for x in log_check)
end

function _run_bp(schedule::String)
    if schedule == "serial"
        return run_serial_belief_propagation!
    elseif schedule == "parallel"
        return run_belief_propagation!
    else
        error("Invalid schedule. Choose either 'serial' or 'parallel'.")
    end
end

function _syndrome_representative(
    y::Vector{Float64},
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    decoding_style::String,
)
    if decoding_style == "syndrome"
        syndrome = (H * y) .% 1
        return G * syndrome
    elseif decoding_style == "received_vector"
        return copy(y)
    else
        error("Invalid decoding style. Choose either 'syndrome' or 'received_vector'.")
    end
end

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
    run_bp! = _run_bp(schedule)

    return @distributed (+) for _ in 1:n_samples
        y = sample_error(σ, tg.nv)
        η = _syndrome_representative(y, H, G, decoding_style)

        bp_result = run_bp!(tg, η, σ, iterations, decoder, search_interval=search_radius)
        dec = hard_decision(bp_result, H)

        if local_search
            λ = abs.(H * bp_result) .% 1.0
            local_search!(η, λ, dec, lsd)
        end

        correction = η - lsd.G * dec
        residual = y - correction

        is_not_logical_error(logical_check, residual) ? 0 : 1
    end
end

function qec_experiment(;
    problem::QECProblem,
    n_samples::Int64,
    params::Dict,
)
    tg = initialize_tanner_graph(problem.H)
    lsd = LocalSearch(
        length(problem.order),
        problem.G,
        problem.order,
        params[:local_search_lll],
        params[:sphere_decoding],
        params[:full_basis],
    )

    results = Int64[]
    for σ in params[:sigmas]
        push!(
            results,
            qec_sample(;
                H = problem.H,
                G = problem.G,
                logical_check = problem.logical_check,
                lsd,
                tg,
                σ,
                n_samples,
                params...,
            ),
        )
    end
    return results
end

function parameter_grid(param_ranges::Dict)
    keys_list = collect(keys(param_ranges))
    values_list = collect(values(param_ranges))

    return [
        Dict(zip(keys_list, combo))
        for combo in Iterators.product(values_list...)
    ]
end

function build_metadata(params::Dict, σ, H; extra...)
    meta = Dict{Symbol,Any}()

    for (key, value) in pairs(params)
        meta[key] = value
    end

    meta[:sigma] = σ
    meta[:nbits] = size(H, 2)

    for (key, value) in extra
        meta[key] = value
    end

    delete!(meta, :sigmas)

    if !meta[:local_search]
        meta[:local_search_lll] = false
        meta[:sphere_decoding] = false
    end

    return meta
end

function run_experiment_grid(;
    path::AbstractString,
    param_ranges::Dict,
    n_samples::Int64,
    repeats::Int64,
    experiment::Function,
)
    for _ in 1:repeats
        for params in parameter_grid(param_ranges)
            results, H = experiment(params, n_samples)

            for (σ, errors) in zip(params[:sigmas], results)
                meta = build_metadata(params, σ, H; d = params[:d])

                add_data!(
                    path;
                    shots = n_samples,
                    errors = errors,
                    decoder = params[:decoder],
                    json_metadata = meta,
                )
            end

            flush(stdout)
        end
    end
end
