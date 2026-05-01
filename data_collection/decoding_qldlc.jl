using Distributed
addprocs(10);
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
@everywhere using LinearAlgebra
using IterTools: product

# For decoding reduced codes generated via:
# /Users/timo/Documents/gkp_ldlc_mwe-main/examples/reorganized/
# use: data_collection/reorganized/decode_generated_codes.jl


@everywhere function is_not_logical_error(G::AbstractMatrix{Float64}, res::Vector{Float64}, eps::Float64=1e-5)
    log_check = G' * res
    return all(abs(x - round(x)) < eps for x in log_check)
end


@everywhere function qec_sample(;
    tg::LatticeDecoder.TannerGraph,
    lsd::LatticeDecoder.LocalSearch,
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical_check::AbstractMatrix{Float64},
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

        dec = hard_decision(bp_result, H)

        if local_search
            λ = abs.(H * bp_result) .% 1.0
            local_search!(η, λ, dec, lsd)
        end

        corr = η - lsd.G * dec
        res = y - corr
        
        is_not_logical_error(logical_check, res) ? 0 : 1
       
    end

    return tot_errors
end


@everywhere function qec_experiment(;
    H::AbstractArray{Float64},
    G::AbstractArray{Float64},
    order::AbstractArray{Int64},
    n_samples::Int64,
    params::Dict,
)

    tg = initialize_tanner_graph(H)
    lsd = LocalSearch(length(order), G, order, params[:local_search_lll], params[:sphere_decoding], params[:full_basis])
    logical_check = inv(H)
    results = []
    for σ in params[:sigmas]
        print("σ = $(σ)\r")
        push!(results, qec_sample(;H=H, G=G, logical_check=logical_check, lsd=lsd, tg=tg, σ=σ, n_samples=n_samples, params...
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





function run_surface_code_experiment(p, n_samples::Int64)
    
    # println(keys(p))
    # Build the code
    code = GKP_Surface_Code(p[:d], false)
    M = code.code
    J = code.J

    H = -M * J
    G = J * inv(M)  # M⟂ = (J M^T)^{-1} <=> M⟂^T = J M^{-1} 


    order   = p[:local_search] ?  append!([2], fill(1, p[:d] - 1)) : [0]
    p[:iterations] = size(H, 2)


    # Run experiments
    results = qec_experiment(;
        H=H,
        G=G,
        order=order,
        n_samples=n_samples,
        params=p,
    )

    return results, H
end


path = "results/surface_code/test_surface_code_decoding_nonCSS.csv"


param_ranges = Dict(
    :search_radius   => [1.0],
    :decoder         => ["lsd"],
    :local_search    => [false],
    :local_search_lll=> [false],
    :schedule        => ["serial"],
    :decoding_style  => ["received_vector"],
    :d               => [3, 5, 7, 9, 11, 13],  #  
    :sigmas          => [[0.4, 0.45, 0.5, 0.55, 0.60, 0.65] ./ sqrt(2π)],  # 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    :sphere_decoding => [false, true],
    :reduced_basis   => [false],
    :full_basis      => [false],
)


keys_list   = collect(keys(param_ranges));
values_list = collect(values(param_ranges));

params_list = [
    Dict(zip(keys_list, combo))
    for combo in product(values_list...)
];

n_samples = 1_000
for r = 1:50
    for p in params_list
        results, H = run_surface_code_experiment(p, n_samples)

        for (σ, res) in zip(p[:sigmas], results)

            meta = build_metadata(p, σ, H;
                d = p[:d],
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





# display(p)
