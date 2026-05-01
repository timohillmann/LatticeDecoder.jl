using Distributed
addprocs(19);
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
@everywhere using LinearAlgebra
@everywhere using NPZ
@everywhere using SparseArrays: sparse, SparseMatrixCSC
using IterTools: product



@everywhere function is_logical_error(M::AbstractMatrix{Float64}, res::Vector{Float64})
    # Note that this check does not perform symplectic multiplication. Thus,
    # it assumes either M <- M * J or that standard multiplication is sufficient.
    # println(round.(M * res, digits=5))
    # log_check = mod.(round.(Int, M * res), 2) .!= 0
    log_check = round.(Mperp * J * res, digits=0) - round.(Mperp * J * res, digits=5) .!= 0.0
    return any(log_check)
end

# @everywhere function is_logical_error(logical::AbstractVecOrMat{Float64}, res::Vector{Float64})
#     log_check = (round.(Int, abs.(2 .* logical * res )) .% 2) .!= 0
#     return any(log_check)
# end


@everywhere function qec_sample(;
    tg::LatticeDecoder.TannerGraph,
    lsd::LatticeDecoder.LocalSearch,
    M::AbstractMatrix{Float64},
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
    q::Int = 2,
    extras...,
)

    if schedule == "serial"
        run_bp! = run_serial_belief_propagation!
    else
        run_bp! = run_belief_propagation!
    end
    
    H_R = inv(G)
    tot_errors = @distributed (+) for idx = 1:n_samples
        y = sample_error(σ, tg.nv) ./ sqrt(q)
        if decoding_style == "syndrome"
            s = (M * y) .% 1
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
        # is_logical_error(logical, res) ? 1 : 0
        is_logical_error(M, res) ? 1 : 0
       
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
        push!(results, qec_sample(;logical=logical, M=H, G=G, lsd=lsd, tg=tg, σ=σ, n_samples=n_samples, params...
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

@everywhere get_p(code_name::String) = parse(Int, match(r"_p(\d+)", code_name).captures[1])

@everywhere function load_code(code_name::String, balance_weights::Bool = false, reduced_basis::Bool=true)

    if reduced_basis
        p = get_p(code_name)
        code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_expanded.npz")

        if balance_weights
            H = code_data["hx"]
            H_CSC = sparse(H)
            LatticeDecoder.balance_weights!(H_CSC)
            H = Matrix(H_CSC / sqrt(p))
            return H, code_data["lx"] / sqrt(p)
        end

        return code_data["hx"] / sqrt(p), code_data["lx"] / sqrt(p)
    
    else
        p = get_p(code_name)
        code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_overcomplete.npz")
        H = code_data["hx"];
        G = code_data["Gz"];
        return H / sqrt(p), G, code_data["lx"] / sqrt(p)
    end

end

function run_experiment(p, n_samples::Int64)
    # Build the code
    if p[:reduced_basis]
        q = get_p(p[:code_name])
        H, logical = load_code(p[:code_name], p[:balance_weights], p[:reduced_basis])
        G = round.(sqrt(q) * inv(H)) / sqrt(q)

    else
        H, G, logical = load_code(p[:code_name], p[:balance_weights], p[:reduced_basis])
    end
    order = p[:local_search] ? append!([2], fill(1, p[:d]-1)) : [0]
    
    # if p[:reduced_basis]
    #     G = round.(sqrt(2) * inv(H)) / sqrt(2)
    # else
    #     q = get_p(p[:code_name])
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


path = "results/quantum_codes/bivariate_bicycle_paper.csv"


param_ranges = Dict(
    :search_radius   => [1.0],
    :decoder         => ["lsd"],
    :local_search    => [true],
    # :code_name      => ["$(2*L^2)_2_$(L)_p2" for L = 3:2:11],
    # :code_name       => ["36_4_6_p7","48_2_6_p7","54_4_8_p7","72_4_8_p7"],  # "84_2_13_p7", "48_4_7_p5", "144_12_12_p2"
    # :code_name       => ["24_8_2_p7","36_8_2_p7","36_8_4_p7","48_8_4_p7","54_8_6_p7","60_8_6_p7", "72_8_6_p7"],
    :code_name       => ["48_4_7_p2"],
    # :code_name       => ["48_4_5_p7","48_4_6_p7","64_4_6_p7","96_8_5_p7","96_8_6_p7","48_2_8_p7"],
    # :code_name       => ["80_4_16_p7", "48_4_5_p7"], # "84_2_10_p7","84_4_10_p7","96_4_9_p7"
    # :code_name       => String["24_2_5_p3_0","32_2_4_p3_0","32_2_4_p3_1","32_2_4_p3_2","32_4_4_p3_0","32_4_4_p3_1","32_4_4_p3_2","36_2_6_p3_0","36_2_6_p3_1","36_2_6_p3_2","36_2_6_p3_3","36_2_6_p3_4","36_4_6_p3_0","36_4_6_p3_1","40_2_4_p3_0","40_2_4_p3_1","40_2_5_p3_0","40_2_5_p3_1","40_2_5_p3_2","40_2_5_p3_3","40_2_5_p3_4","40_2_5_p3_5","40_2_5_p3_6","40_2_5_p3_7","40_2_7_p3_0","40_2_7_p3_1","40_2_8_p3_0", "48_2_8_p3_0","48_2_8_p3_1","48_2_8_p3_2","48_2_8_p3_3","48_2_8_p3_4","48_2_8_p3_5","48_2_8_p3_6","48_2_8_p3_7"],
    # :code_name          => ["56_2_8_p3_0","56_2_8_p3_1","56_2_8_p3_2","60_2_8_p3_0","60_2_8_p3_1","60_2_8_p3_2","60_2_8_p3_3","60_2_8_p3_4","60_2_8_p3_5","60_2_8_p3_6","60_2_8_p3_7","60_2_8_p3_8","64_2_8_p3_0","64_2_8_p3_1","64_2_8_p3_10","64_2_8_p3_11","64_2_8_p3_12","64_2_8_p3_2","64_2_8_p3_3","64_2_8_p3_4","64_2_8_p3_5","64_2_8_p3_6","64_2_8_p3_7","64_2_8_p3_8","64_2_8_p3_9","64_8_4_p3_0","64_8_4_p3_1","72_2_8_p3_0","72_2_8_p3_1","72_8_6_p3_0","72_8_6_p3_1","80_2_8_p3_0","80_2_8_p3_1","80_2_8_p3_2","80_2_8_p3_3","80_2_8_p3_4","80_4_8_p3_0","80_8_5_p3_0","84_2_8_p3_0","84_2_8_p3_1","84_2_8_p3_2","84_2_8_p3_3","84_2_8_p3_4","84_2_8_p3_5","84_2_8_p3_6","88_2_8_p3_0","96_2_8_p3_0","96_2_8_p3_1","96_2_8_p3_10","96_2_8_p3_11","96_2_8_p3_12","96_2_8_p3_13","96_2_8_p3_14","96_2_8_p3_15","96_2_8_p3_2","96_2_8_p3_3","96_2_8_p3_4","96_2_8_p3_5","96_2_8_p3_6","96_2_8_p3_7","96_2_8_p3_8","96_2_8_p3_9","96_4_8_p3_0","96_4_8_p3_1","96_4_8_p3_2","96_4_8_p3_3","96_4_8_p3_4","96_8_5_p3_0"], 
    # :code_name       => ["24_8_2_p2_0","24_8_2_p2_1","36_12_2_p2_0","36_12_2_p2_1","36_4_2_p2_0","36_4_2_p2_1","36_4_2_p2_10","36_4_2_p2_11","36_4_2_p2_12","36_4_2_p2_13","36_4_2_p2_14","36_4_2_p2_15","36_4_2_p2_16","36_4_2_p2_18","36_4_2_p2_19","36_4_2_p2_2","36_4_2_p2_20","36_4_2_p2_21","36_4_2_p2_22","36_4_2_p2_3","36_4_2_p2_4","36_4_2_p2_5","36_4_2_p2_6","36_4_2_p2_7","36_4_2_p2_8","36_4_2_p2_9","36_4_3_p2_0","36_4_3_p2_1","36_4_3_p2_2","36_4_3_p2_3","36_8_2_p2_0","36_8_3_p2_0","36_8_3_p2_1","36_8_3_p2_2","36_8_3_p2_3","36_8_3_p2_4","36_8_3_p2_5","36_8_3_p2_6","36_8_3_p2_7","36_8_3_p2_8","42_6_3_p2_0","42_6_3_p2_1","42_6_3_p2_10","42_6_3_p2_11","42_6_3_p2_12","42_6_3_p2_13","42_6_3_p2_14","42_6_3_p2_15","42_6_3_p2_2","42_6_3_p2_3","42_6_3_p2_4","42_6_3_p2_5","42_6_3_p2_6","42_6_3_p2_7","42_6_3_p2_8","42_6_3_p2_9","48_16_2_p2_0","48_16_2_p2_1","48_16_2_p2_2","48_16_2_p2_3","48_16_2_p2_4","48_16_2_p2_5","48_4_2_p2_0","48_4_2_p2_1","48_4_2_p2_2","48_4_2_p2_3","48_4_2_p2_4","48_4_2_p2_5","48_4_2_p2_6","48_4_2_p2_7","48_4_2_p2_8","48_4_2_p2_9","48_4_4_p2_0","48_4_4_p2_1","48_4_4_p2_2","48_4_4_p2_3","48_4_4_p2_4","48_4_4_p2_5","48_4_4_p2_6","48_4_4_p2_7","48_4_4_p2_8","48_8_2_p2_0","48_8_2_p2_1","48_8_2_p2_10","48_8_2_p2_11","48_8_2_p2_12","48_8_2_p2_13","48_8_2_p2_14","48_8_2_p2_15","48_8_2_p2_16","48_8_2_p2_17","48_8_2_p2_18","48_8_2_p2_19","48_8_2_p2_2","48_8_2_p2_20","48_8_2_p2_3","48_8_2_p2_4","48_8_2_p2_5","48_8_2_p2_6","48_8_2_p2_7","48_8_2_p2_8","48_8_2_p2_9","48_8_3_p2_0","48_8_3_p2_1","48_8_3_p2_2","48_8_3_p2_3","48_8_4_p2_0","48_8_4_p2_1","48_8_4_p2_2","48_8_4_p2_3","48_8_4_p2_4","48_8_4_p2_5","54_12_2_p2_0","54_12_3_p2_0","54_12_3_p2_1","54_12_3_p2_2","54_12_3_p2_3","54_12_3_p2_4","54_12_3_p2_5","54_12_3_p2_6","54_12_3_p2_7","54_12_3_p2_8","54_4_3_p2_0","54_4_3_p2_1","54_4_3_p2_10","54_4_3_p2_11","54_4_3_p2_12","54_4_3_p2_13","54_4_3_p2_14","54_4_3_p2_15","54_4_3_p2_16","54_4_3_p2_17","54_4_3_p2_18","54_4_3_p2_19","54_4_3_p2_2","54_4_3_p2_20","54_4_3_p2_21","54_4_3_p2_22","54_4_3_p2_23","54_4_3_p2_24","54_4_3_p2_25","54_4_3_p2_26","54_4_3_p2_27","54_4_3_p2_28","54_4_3_p2_29","54_4_3_p2_3","54_4_3_p2_30","54_4_3_p2_31","54_4_3_p2_32","54_4_3_p2_33","54_4_3_p2_34","54_4_3_p2_35","54_4_3_p2_36","54_4_3_p2_37","54_4_3_p2_38","54_4_3_p2_39","54_4_3_p2_4","54_4_3_p2_40","54_4_3_p2_41","54_4_3_p2_42","54_4_3_p2_43","54_4_3_p2_44","54_4_3_p2_45","54_4_3_p2_46","54_4_3_p2_47","54_4_3_p2_48","54_4_3_p2_49","54_4_3_p2_5","54_4_3_p2_50","54_4_3_p2_51","54_4_3_p2_52","54_4_3_p2_53","54_4_3_p2_54","54_4_3_p2_55","54_4_3_p2_56","54_4_3_p2_57","54_4_3_p2_58","54_4_3_p2_59","54_4_3_p2_6","54_4_3_p2_60","54_4_3_p2_61","54_4_3_p2_62","54_4_3_p2_63","54_4_3_p2_64","54_4_3_p2_65","54_4_3_p2_66","54_4_3_p2_7","54_4_3_p2_8","54_4_3_p2_9","56_12_2_p2_0","56_12_2_p2_1","56_12_2_p2_10","56_12_2_p2_11","56_12_2_p2_12","56_12_2_p2_13","56_12_2_p2_14","56_12_2_p2_15","56_12_2_p2_2","56_12_2_p2_3","56_12_2_p2_4","56_12_2_p2_5","56_12_2_p2_6","56_12_2_p2_7","56_12_2_p2_8","56_12_2_p2_9","56_6_2_p2_0","56_6_2_p2_1","56_6_2_p2_10","56_6_2_p2_11","56_6_2_p2_12","56_6_2_p2_13","56_6_2_p2_14","56_6_2_p2_15","56_6_2_p2_16","56_6_2_p2_17","56_6_2_p2_18","56_6_2_p2_19","56_6_2_p2_2","56_6_2_p2_3","56_6_2_p2_4","56_6_2_p2_5","56_6_2_p2_6","56_6_2_p2_7","56_6_2_p2_8","56_6_2_p2_9","56_6_4_p2_0","56_6_4_p2_1","56_6_4_p2_10","56_6_4_p2_11","56_6_4_p2_12","56_6_4_p2_13","56_6_4_p2_14","56_6_4_p2_15","56_6_4_p2_16","56_6_4_p2_17","56_6_4_p2_18","56_6_4_p2_19","56_6_4_p2_2","56_6_4_p2_20","56_6_4_p2_21","56_6_4_p2_22","56_6_4_p2_23","56_6_4_p2_24","56_6_4_p2_25","56_6_4_p2_26","56_6_4_p2_27","56_6_4_p2_28","56_6_4_p2_29","56_6_4_p2_3","56_6_4_p2_30","56_6_4_p2_31","56_6_4_p2_32","56_6_4_p2_33","56_6_4_p2_34","56_6_4_p2_35","56_6_4_p2_4","56_6_4_p2_5","56_6_4_p2_6","56_6_4_p2_7","56_6_4_p2_8","56_6_4_p2_9","60_20_2_p2_0","60_20_2_p2_1","60_4_2_p2_0","60_4_2_p2_1","60_4_2_p2_10","60_4_2_p2_11","60_4_2_p2_12","60_4_2_p2_13","60_4_2_p2_14","60_4_2_p2_15","60_4_2_p2_16","60_4_2_p2_17","60_4_2_p2_18","60_4_2_p2_19","60_4_2_p2_2","60_4_2_p2_20","60_4_2_p2_21","60_4_2_p2_22","60_4_2_p2_23","60_4_2_p2_24","60_4_2_p2_25","60_4_2_p2_26","60_4_2_p2_27","60_4_2_p2_28","60_4_2_p2_29","60_4_2_p2_3","60_4_2_p2_4","60_4_2_p2_5","60_4_2_p2_6","60_4_2_p2_7","60_4_2_p2_8","60_4_2_p2_9","60_4_4_p2_0","60_4_4_p2_1","60_4_4_p2_2","60_4_5_p2_0","60_4_5_p2_1","60_4_5_p2_10","60_4_5_p2_11","60_4_5_p2_12","60_4_5_p2_13","60_4_5_p2_14","60_4_5_p2_15","60_4_5_p2_16","60_4_5_p2_17","60_4_5_p2_18","60_4_5_p2_19","60_4_5_p2_2","60_4_5_p2_20","60_4_5_p2_21","60_4_5_p2_22","60_4_5_p2_23","60_4_5_p2_24"],
    :balance_weights => [true],
    :local_search_lll=> [true],
    :schedule        => ["serial"],
    :decoding_style  => ["received_vector"],
    # :sigmas          => [[0.3] ./ sqrt(2π)],
    :sigmas          => [ [0.35, 0.4] ./ sqrt(2π)],  # , 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6
    # :sigmas          => [ [0.15, .16, .17, .18, .19, 0.2] ./ sqrt(2π)],
    :sphere_decoding => [false],
    :reduced_basis   => [true],
    :iterations      => [50],
)


keys_list   = collect(keys(param_ranges));
values_list = collect(values(param_ranges));

params_list = [
    Dict(zip(keys_list, combo))
    for combo in product(values_list...)
];

for r in 1:10
    n_samples = 50_000
    jj=1
    tot_settings = length(params_list)
    for p in params_list
        println("Running setting $(jj) / $(tot_settings) \r")
        flush(stdout)
        p[:d] = parse(Int, split(p[:code_name], "_")[3])
        p[:q] = get_p(p[:code_name])
        try
            results, M = run_experiment(p, n_samples)

            for (σ, res) in zip(p[:sigmas], results)

                meta = build_metadata(p, σ, M;
                )

                add_data!(
                    path;
                    shots = n_samples,
                    errors = res,
                    decoder = string(p[:decoder]),
                    json_metadata = meta
                )
            end
        catch 
            println("$(p[:code_name]) didnt work.")
        end
        jj+=1
    end
end

