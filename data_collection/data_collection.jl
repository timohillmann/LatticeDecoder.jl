using Distributed
addprocs(18);
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
@everywhere using LinearAlgebra
using IterTools: product


@everywhere function is_logical_error(logical::Vector{Float64}, res::Vector{Float64})
    log_check = (round.(Int, abs.(2 .*logical' * res)) .% 2) .!= 0
    return any(log_check)
end

@everywhere function is_in_code_space_rep_code(res::Vector{Float64})
    
    # check is residual is all zeros or all ones
    res *= sqrt(2)
    res .= abs.(res)
    res .= round.(res)
    res .= res .% 2
    
    # println("res after scaling: $(res)")

    eval = all(x -> isapprox(x, 0.0; atol=1e-3) || isapprox(x, 1.0; atol=1e-3), res)
    # println(eval)
    return eval
end

@everywhere function qec_sample(;
    tg::LatticeDecoder.TannerGraph,
    lsd::LatticeDecoder.LocalSearch,
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical::Vector{Float64},
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
    results = [
        qec_sample(;logical=logical, H=H, G=G, lsd=lsd, tg=tg, σ=σ, n_samples=n_samples, params...
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



function run_experiment(p, n_samples::Int64)
    
    # println(keys(p))
    # Build the code
    code = GKP_Rep_Code(p[:n], false, p[:reduced_basis])
    logical = Vector(code.logical[1:p[:n]])
    order   = p[:local_search] ?  append!([2], fill(1, p[:n] ÷ 2)) : [0]

    H = code.code[p[:n]+1:end, p[:n]+1:end]

    # H[1, 1] = 1 ./ sqrt(2)
    # H[1, 2] = -1 ./ sqrt(2)
    # H[1, end] = 1 ./ sqrt(2)
    # display(H)

    if p[:reduced_basis]
        G = round.(sqrt(2) * inv(H)) / sqrt(2)
    else
        H_R, _ = hnfr(round.(Int, sqrt(2) * H))
        H_R = H_R[1:p[:n], :]  # drop zero rows
        G = inv(H_R / sqrt(2))
        Gp, _ = lll(G')
        G = Matrix(Gp')
    end
    # H[1, 1:2] = [3, 1] / sqrt(2)
    # H[1, 1] = 3.0 / sqrt(2)
    # H[1, 2] = 0.0 / sqrt(2)
    # H[1, end] = -1.0 / sqrt(2)

    # display(H)

    p[:iterations] = 2 * p[:n]

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


path = "results/quantum_rep_code/rep_code_standard_qctip.csv"


param_ranges = Dict(
    :search_radius   => [1.0],
    :decoder         => ["lsd"],
    :local_search    => [false],
    :local_search_lll=> [true],
    :schedule        => ["serial"],
    :decoding_style  => ["received_vector"],
    :n               => [7, 9, 11, 13],  #  
    :sigmas          => [[0.4] ./ sqrt(2π)],  # 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    :sphere_decoding => [false],
    :reduced_basis   => [true],
)


keys_list   = collect(keys(param_ranges));
values_list = collect(values(param_ranges));

params_list = [
    Dict(zip(keys_list, combo))
    for combo in product(values_list...)
];

n_samples = 200_000
for r = 1:50
    for p in params_list
        results, H = run_experiment(p, n_samples)

        for (σ, res) in zip(p[:sigmas], results)

            meta = build_metadata(p, σ, H;
                d = p[:n],
            )

            add_data!(
                path;
                shots = n_samples,
                errors = res,
                decoder = p[:decoder],
                json_metadata = meta
            )
        end

        flush(stdout)
    end
end



