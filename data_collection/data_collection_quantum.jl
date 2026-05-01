using Distributed
addprocs(14);
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
        run_bp! = run_serial_belief_propagation!
    else
        run_bp! = run_belief_propagation!
    end
    
    H_R = inv(G)
    tot_errors = @distributed (+) for idx = 1:n_samples
        y = sample_error(σ, tg.nv)
        if decoding_style == "syndrome"
            s = (H * y) .% 1
            η = G * s
        elseif decoding_style == "received_vector"
            η = copy(y)
        else
            error("Invalid decoding style. Choose either 'syndrome' or 'received_vector'.")
        end

        bp_result = run_bp!(tg, η, σ, iterations, decoder, search_interval=search_radius)
        dec = hard_decision(bp_result, H_R)

        if local_search
            λ = abs.(H_R * bp_result) .% 1.0
            local_search!(η, λ, dec, lsd)
        end

        corr = η - lsd.G * dec
        res = y - corr

        # @assert is_in_code_space_rep_code(res)  # optional check

        is_logical_error(logical, res) ? 1 : 0
       
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
    results = []
    for σ in params[:sigmas]
        print("σ = $(σ)\r")
        push!(results, qec_sample(;logical=logical, H=H, G=G, lsd=lsd, tg=tg, σ=σ, n_samples=n_samples, params...
        ))
    end
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

@everywhere function load_code(code_name::String, balance_weights::Bool = false, reduced_basis::Bool=true)

    if reduced_basis
        p = parse(Int, code_name[end])
        code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_expanded.npz")

        if balance_weights
            H = code_data["hx"]
            H_CSC = sparse(H)
            LatticeDecoder.balance_weights!(H_CSC)
            H = Matrix(H_CSC / sqrt(p))
            return H, code_data["lz"] / sqrt(p)
        end

        return code_data["hx"] / sqrt(p), code_data["lx"] / sqrt(p)
    
    else
        p = parse(Int, code_name[end])
        code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_overcomplete.npz")
        H = code_data["hx"];
        G = code_data["Gz"];
        return H / sqrt(p), G, code_data["lz"] / sqrt(p)
    end

end

function run_experiment(p, n_samples::Int64)
    

    # Build the code
    if p[:reduced_basis]
        H, logical = load_code(p[:code_name], p[:balance_weights], p[:reduced_basis])
        G = round.(sqrt(2) * inv(H)) / sqrt(2)

    else
        H, G, logical = load_code(p[:code_name], p[:balance_weights], p[:reduced_basis])
    end
    println("Code loaded.")
    order = p[:local_search] ? append!([2], fill(1, p[:d]-1)) : [0]
    
    # if p[:reduced_basis]
    #     G = round.(sqrt(2) * inv(H)) / sqrt(2)
    # else
    #     q = parse(Int, p[:code_name][end])
    #     n = size(H, 2)
    #     H_R, _ = hnfr(round.(Int128, sqrt(q) * H))
    #     println("Code reduced.")
    #     H_R = H_R[1:n, :]  # drop zero rows
    #     G = inv(H_R / sqrt(q))
    #     println("Generator initialized.")
    #     Gp, _ = lll(G')
    #     println("Generator reduced.")
    #     G = Float64.(Matrix(Gp'))
    # end



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


path = "results/quantum_codes/bivariate_bicycle_w5.csv"


param_ranges = Dict(
    :search_radius   => [1.0],
    :decoder         => [3],
    :local_search    => [true],
    :code_name       => ["30_4_5_p2", "48_4_7_p2"],  # "84_2_13_p7", "48_4_7_p5", "144_12_12_p2"
    :balance_weights => [true],
    :local_search_lll=> [true],
    :schedule        => ["serial"],
    :decoding_style  => ["received_vector"],
    :sigmas          => [ [0.4, 0.45, 0.50] ./ sqrt(2π)],
    :sphere_decoding => [false],
    :reduced_basis   => [true],
    :iterations      => [30],
)


keys_list   = collect(keys(param_ranges));
values_list = collect(values(param_ranges));

params_list = [
    Dict(zip(keys_list, combo))
    for combo in product(values_list...)
];

for r in 1:5
    n_samples = 10_000
    jj=1
    tot_settings = length(params_list)
    for p in params_list
        println("Running setting $(jj) / $(tot_settings) \r")
        
        p[:d] = parse(Int, split(p[:code_name], "_")[3])
        results, H = run_experiment(p, n_samples)

        for (σ, res) in zip(p[:sigmas], results)

            meta = build_metadata(p, σ, H;
            )

            add_data!(
                path;
                shots = n_samples,
                errors = res,
                decoder = string(p[:decoder]),
                json_metadata = meta
            )
        end
        jj+=1
        flush(stdout)
    end
end

