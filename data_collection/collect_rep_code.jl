using Distributed
# addprocs(18);
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
@everywhere using LinearAlgebra

println("Number of workers: ", nworkers())

@everywhere function qec_sample(
    tg::LatticeDecoder.TannerGraph,
    lsd::LatticeDecoder.LocalSearch,
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    logical::Vector{Float64},
    σ::Float64,
    n_samples;
    schedule::String = "serial",
    iterations::Int = tg.nv,
    decoder::Union{String, Int64} = "lsd",
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
            λ = abs.(H * bp_result) .% 1.0
            local_search!(η, λ, dec, lsd)
        end

        corr = η - lsd.G * dec
        res = y - corr


        # check for quantum error 
        # if the error is not in the code space, then it is a quantum error
        # rnd_res = round.(Int, abs.(res * sqrt(2))) .% 2

        
        # num_faults = sum(rnd_res)
        # if num_faults == tg.nv
        #     return 1
        # elseif num_faults == 0
        #     return 0
        # else
        #     error("Out of code space error: $(rnd_res)")
        # end

        # printstyled("Log Com. Check: $((round.(Int, abs.(logical' * res))))", color=:green)

        # println("res = $(res)")
        # println("logical = $(logical)")
        # println("Logical' * res = $(round.(Int, abs.(2 .*logical' * res)) .% 2)")

        log_check = (round.(Int, abs.(2 .*logical' * res)) .% 2) .!= 0
        # println(log_check)
        if any(log_check)
            # println("Logical error occurred.")
            1
        else
            # println("No logical error.")
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
    sphere_decoding,
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
            local_search=local_search,
            decoding_style=decoding_style,
            search_radius=search_radius
        ) for σ in sigmas
    ]
    return results
end


global result_dict = Dict()
reduced_decoding = true
sphere_decoding = true
# path = "results/quantum_rep_code/standard_reduced_rep_code_ls.csv"
# path = "results/quantum_rep_code/standard_reduced_check_matrix_ls.csv"
path = "results/quantum_rep_code/test_standard.csv"
# path = "results/quantum_rep_code/balanced_second_last_reduced_check_matrix_ls_sd.csv"

# radii = 0.60:0.05:0.8
# radii = 1.20:0.025:1.3
# radii = 0.80:0.05:1.3

# radii = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.10, 1.15, 1.2, 1.225, 1.25, 1.275, 1.300]

n_samples = 10_000
# 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42
for r = 1:10
    for radius in [1.0] # radii # [sqrt(1/3), sqrt(1/2), sqrt(2/3), sqrt(3/4), sqrt(1), sqrt(4/3), sqrt(5/4), 1.1, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32]
        # printstyled("Running experiments for ϵ = $radius, \t r = $r \n", color=:red)
        for decoder in ["lsd", "nearest", 3, 2]
            for local_search in [false]
                for local_search_lll in [true]
                    local_search_lll = local_search ? local_search_lll : false
                    sphere_decoding = local_search ? sphere_decoding : false
                    for schedule in ["parallel", "serial"]
                        for dec_style in ["received_vector"]
                            # for n in [3]
                            # Inline progress print for radius/n/r:
                            # print("\rProgress: r = $r, radius = $radius", end="", flush=true)
                            for n in [3, 5, 7, 9, 11, 13]
                                printstyled("Running $(decoder) for ϵ = $radius, n = $n, r = $r\n", color=:red)
                                code = GKP_Rep_Code(n, false, reduced_decoding)
                                logical = Vector(code.logical[1:n])
                                order = local_search ? fill(1, n ÷ 2) : [0]
                                reduced_basis = true
                                search_radius = radius
                                iterations = 2 * n

                                H = code.code[n+1:end, n+1:end]
                                # display(H)
                                # H[1, 1:2] = [1, -1] / sqrt(2)
                                # H[1, 1] = 1.0 / sqrt(2)
                                # H[1, 2] = 0.0 / sqrt(2)
                                # H[1, end] = 1.0 / sqrt(2)

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

                                sigmas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] ./ sqrt(2 * pi) # [0.4, 0.5, 0.6] ./ sqrt(2 * pi)
                                # sigmas = [0.8, 0.9, 1.0] ./ sqrt(2 * pi)
                                results = qec_experiment(logical, H, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations,
                                    decoder, dec_style, search_radius, local_search_lll, sphere_decoding)

                                for (σ, res) in zip(sigmas, results)
                                    json_data = metadata(
                                        code="rep_code",
                                        schedule=schedule,
                                        decoder=string(decoder),  # "lsd",
                                        d=n,
                                        local_seach_order=order,
                                        local_search=local_search,
                                        local_search_lll=local_search_lll,
                                        reduced_basis=reduced_basis,
                                        sigma=σ,
                                        search_radius=search_radius,
                                        iterations=iterations,
                                        decoding_style=dec_style,
                                        sphere_decoding=sphere_decoding,
                                        nbits=size(H, 2),
                                    )
                                    # we add shots = n_samples * nbits to plots the correct statistics, the symbol error rate.
                                    add_data!(path, shots=n_samples, errors=res, decoder=string(decoder), json_metadata=json_data)
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
