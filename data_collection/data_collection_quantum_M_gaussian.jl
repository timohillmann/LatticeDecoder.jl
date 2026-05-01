using Distributed
addprocs(19);
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
@everywhere using LinearAlgebra
@everywhere using NPZ
@everywhere using SparseArrays: sparse, SparseMatrixCSC
using IterTools: product


@everywhere function is_logical_error(logical::AbstractVecOrMat{Float64}, res::Vector{Float64})
    log_check = (round.(Int, abs.(2 .* logical * res )) .% 2) .!= 0
    return any(log_check)
end

@everywhere function qec_sample(;
    dec::LDLCDecoder,
    tg::LatticeDecoder.TannerGraph,
    lsd::LatticeDecoder.LocalSearch,
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical::AbstractVecOrMat{Float64},
    σ::Float64,
    n_samples::Int,
    schedule::String = "serial",
    iterations::Int = tg.nv,
    decoder::Union{String, Int64} = "lsd",
    decoding_style::String = "syndrome",
    search_radius::Float64 = 1.5,
    local_search::Bool = false,
    extras...,
)

    if schedule == "serial"
        run_bp! = run_decoder_serial!
    else
        run_bp! = run_decoder_parallel!
    end

    tot_errors = 0
    for idx = 1:n_samples
        y = sample_error(σ, tg.nv)
        if decoding_style == "syndrome"
            s = (H * y) .% 1
            η = G * s
        elseif decoding_style == "received_vector"
            η = copy(y)
        else
            error("Invalid decoding style. Choose either 'syndrome' or 'received_vector'.")
        end

        bp_result = run_bp!(dec, η, σ, iterations)
        dec = hard_decision(bp_result, H)

        if local_search
            λ = abs.(H * bp_result) .% 1.0
            local_search!(η, λ, dec, lsd)
        end

        corr = η - lsd.G * dec
        res = y - corr

        # @assert is_in_code_space_rep_code(res)  # optional check

        tot_errors += is_logical_error(logical, res) ? 1 : 0
       
    end

    return tot_errors
end


@everywhere function qec_experiment(;
    logical::AbstractArray{Float64},
    H::AbstractArray{Float64},
    G::AbstractArray{Float64},
    order::AbstractArray{Int64},
    n_samples::Int64,
    params::Dict,
)
    tg = initialize_tanner_graph(H)
    lsd = LocalSearch(length(order), G, order, params[:local_search_lll], params[:sphere_decoding])
    dec = LDLCDecoder(tg, params[:decoder])
    results = [
        qec_sample(;logical=logical, H=H, G=G, lsd=lsd, tg=tg, σ=σ, n_samples=n_samples, dec=dec, params...
        ) for σ in params[:sigmas]
    ]
    return results
end

function build_metadata(p, σ, H; extra...)
    meta = Dict{Symbol,Any}()
    # include all parameters in p
    for (k,v) in pairs(p)
        meta[k] = v
    end

    # include sigma as part of metadata
    meta[:sigma] = σ
    meta[:nbits] = size(H,2)

    # add arbitrary runtime extras
    for (k,v) in extra
        meta[k] = v
    end

    delete!(meta, :sigmas)

    # logic for overwriting wrong keywords
    if !(meta[:local_search])
        meta[:local_search_lll] = false
        meta[:sphere_decoding] = false
    end

    return meta
end

@everywhere function load_code(code_name::String, balance_weights::Bool = false)

    p = parse(Int, code_name[end])
    code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_expanded.npz")

    if balance_weights
        H = code_data["hx"]
        H_CSC = sparse(H)
        LatticeDecoder.balance_weights!(H_CSC)
        H = Matrix(H_CSC / sqrt(2))
        return H, code_data["lz"] / sqrt(p)
    end

    return code_data["hx"] / sqrt(p), code_data["lx"] / sqrt(p)
end

function run_experiment(p, n_samples::Int64)
    
    # println(keys(p))
    # Build the code
    H, logical = load_code(p[:code_name], p[:balance_weights])
    order = p[:local_search] ? append!([2], fill(1, p[:d]-1)) : [0]
    G = round.(sqrt(2) * inv(H)) / sqrt(2)


    # Run experiments
    results = qec_experiment(;
        logical=logical,
        H=H,
        G=G,
        order=order,
        n_samples=n_samples,
        params=p,
    )

    return results, H
end


path = "results/quantum_codes/bivariate_bicycle_balanced.csv"


param_ranges = Dict(
    :search_radius   => [1.0],
    :decoder         => [2],
    :local_search    => [true],
    :code_name       => ["30_4_5_p2", "48_4_7_p2"],
    :balance_weights => [true],
    :local_search_lll=> [true],
    :schedule        => ["serial"],
    :dec_style       => ["received_vector"],
    :sigmas          => [ [0.3, 0.35, 0.4, 0.45, 0.5, 0.55] ./ sqrt(2π)],
    :sphere_decoding => [false],
    :iterations      => [50]
)


keys_list   = collect(keys(param_ranges));
values_list = collect(values(param_ranges));

params_list = [
    Dict(zip(keys_list, combo))
    for combo in product(values_list...)
];

for r in 1:500
    n_samples = 10
    tot_settings = length(params_list)
    for p in params_list
        # println("Running setting $(jj) / $(tot_settings) \r")
        
        p[:d] = parse(Int, split(p[:code_name], "_")[3])
        results, H = run_experiment(p, n_samples)

        for (σ, res) in zip(p[:sigmas], results)

            meta = build_metadata(p, σ, H;
                reduced_basis = true,
            )

            add_data!(
                path;
                shots = n_samples,
                errors = res,
                decoder = string(p[:decoder]),
                json_metadata = meta
            )
        end
        flush(stdout)
    end
end

