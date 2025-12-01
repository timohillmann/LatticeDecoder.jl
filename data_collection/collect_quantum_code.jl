using Distributed
addprocs(9)
@everywhere using NPZ
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
@everywhere using SparseArrays: sparse, SparseMatrixCSC



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

    return code_data["hx"] / sqrt(p), code_data["lz"] / sqrt(p)

    # if code_name == "30_4_5_p2"
    #     code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/30_4_5_p2_expanded.npz")
    #     return code_data["hx"] / sqrt(2), code_data["lz"] / sqrt(2)
    
    # elseif code_name == "48_4_7_p2"
    #     code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/48_4_7_p2_expanded.npz")
    #     return code_data["hx"] / sqrt(2), code_data["lz"] / sqrt(2)

    # elseif code_name == "30_4_5_p5"
    #     code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/30_4_5_p5_expanded.npz")
    #     return code_data["hx"] / sqrt(5), code_data["lz"] / sqrt(5)
    
    
    # elseif code_name == "48_4_7_p5"
    #     code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/48_4_7_p5_expanded.npz")
    #     return code_data["hx"] / sqrt(5), code_data["lz"] / sqrt(5)
    # end
end


@everywhere function qec_sample(
    tg::LatticeDecoder.TannerGraph,
    lsd::LatticeDecoder.LocalSearch,
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical::AbstractMatrix{Float64},
    σ::Float64,
    n_samples;
    schedule::String = "serial",
    iterations::Int = tg.nv,
    decoder::String = "lsd",
    decoding_style::String = "syndrome",
    search_radius::Float64 = 1.5,
    local_search::Bool = false,
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
            λ = abs(H * bp_result) .% 1.0
            local_search!(η, λ, dec, lsd)
        end

        corr = η - dec
        res = y - corr


        # check for quantum error 
        # if the error is not in the code space, then it is a quantum error
        # rnd_res = round.(Int, abs.(res * sqrt(2))) .% 2
        # if sum(rnd_res) != 0
        #     return 1
        # else
        #     return 0
        # end

        # printstyled("Log Com. Check: $((round.(Int, abs.(logical * res)) .% 2))\n", color=:green)
        
        log_check = (((round.(Int, abs.(logical * res)) .% 2)))
        has_logical_error = any(x != 0 for x in log_check)

        if any.(has_logical_error)
            1
        else
            0
        end
       

        # _errs = count_symbol_errors(res)
        # if _errs > 0
        #     _errs
        # else
        #     0
        # end

    end

    return tot_errors
end


@everywhere function qec_experiment(
    logical,
    H,
    G,
    sigmas,
    n_samples,
    order,
    local_search,
    schedule,
    reduced_basis,
    iterations,
    decoder,
    decoding_style,
    search_radius,
    local_search_lll,
    sphere_decoding
)
    tg = initialize_tanner_graph(H)
    # lsd = LatticeStatisticsDecoding(order, G, reduced_basis)
    lsd = LocalSearch(length(order), G, order, local_search_lll, sphere_decoding)
    results = [
        qec_sample(
            tg,
            lsd,
            H,
            G,
            logical,
            σ,
            n_samples;
            schedule=schedule,
            iterations=iterations,
            decoder=decoder,
            decoding_style=decoding_style,
            search_radius=search_radius
        ) for σ in sigmas
    ]
    return results
end


global result_dict = Dict()
reduced_decoding = true
reduced_basis = true
balance_weights = true
sphere_decoding = false
# radii = 0.5:0.05:1.4
radii = 0.5:0.05:1.2
# radii = [1.0]
path = "results/quantum_codes/small_bb_codes_balanced_radius.csv"
for r = 1:10
    for radius in radii # [sqrt(1/3), sqrt(1/2), sqrt(2/3), sqrt(3/4), sqrt(1), sqrt(4/3), sqrt(5/4), 1.1, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32]
        # printstyled("Running experiments for ϵ = $radius, \t r = $r \n", color=:red)
        for decoder in ["nearest"]
            for local_search in [true]
                for local_search_lll in [true]
                    for schedule in ["serial"]
                        for dec_style in ["received_vector"]
                            # for n in [3]
                            # Inline progress print for radius/n/r:
                            # print("\rProgress: r = $r, radius = $radius", end="", flush=true)
                            for code in ["18_4_3_p2"]
                                printstyled("Running experiment for $(code), ϵ = $radius, r = $r\n", color=:red)
                                H, logical = load_code(code, balance_weights)
                                order = local_search ? [2, 1, 1] : [0]
                                search_radius = radius
                                iterations = 50

                                # H = code.code[n+1:end, n+1:end]
                                # H[1, 1:2] = [1, -1] / sqrt(2)
                                # H[1, 1] = 1.0 / sqrt(2)
                                # H[1, 2] = 0.0 / sqrt(2)
                                # H[1, end-1] = -1.0 / sqrt(2)

                                # display(H)

                                

                                if reduced_decoding
                                    G = round.(sqrt(2) * inv(H)) / sqrt(2)
                                else
                                    H_R, _ = hnfr(round.(Int, sqrt(2) * H))
                                    H_R = H_R[1:n, :]
                                    # drop zero rows
                                    G = inv(H_R / sqrt(2))
                                    G, _ = lll(G')
                                    G = G'
                                    # TODO: G <- lll(G)
                                end

                                sigmas = [0.3] ./ sqrt(2 * pi)
                                # sigmas = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65] ./ sqrt(2 * pi) # [0.4, 0.5, 0.6] ./ sqrt(2 * pi)
                                # sigmas = [0.25, 0.2, 0.3, 0.35] ./ sqrt(2 * pi)
                                # sigmas = [0.2] ./ sqrt(2 * pi)
                                n_samples = 10_000

                                results = qec_experiment(logical, H, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations,
                                    decoder, dec_style, search_radius, local_search_lll, sphere_decoding)

                                for (σ, res) in zip(sigmas, results)
                                    json_data = metadata(
                                        code=code,
                                        schedule=schedule,
                                        decoder=decoder,  # "lsd",
                                        d=parse(Int, split(code, "_")[3]),
                                        local_seach_order=order,
                                        local_search=local_search,
                                        local_search_lll=local_search_lll,
                                        reduced_basis=reduced_basis,
                                        sigma=σ,
                                        search_radius=search_radius,
                                        iterations=iterations,
                                        decoding_style=dec_style,
                                        balance_weights=balance_weights,
                                        nbits=size(H, 2) // 2,
                                        sphere_decoding=sphere_decoding
                                    )
                                    # we add shots = n_samples * nbits to plots the correct statistics, the symbol error rate.
                                    add_data!(path, shots=n_samples, errors=res, decoder=decoder, json_metadata=json_data)
                                    flush(stdout)
                                end
                                # save results to file
                                # result_dict[n] = results
                            end
                        end
                    end
                end
            end
        end
    end
end