using Distributed
addprocs(1);
@everywhere using LatticeDecoder


@everywhere function qec_sample(tg, lsd, H, G, logical, σ, n_samples, local_search::Bool=true, schedule::String="serial", iterations::Int=tg.nv, decoder::String="lsd",
    decoding_style::String="syndrome", search_radius::Float64=1.5)
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
        # bp_result = run_serial_belief_propagation!(tg, y, σ, tg.nv)
        # bp_result = run_belief_propagation!(tg, y, σ, tg.nv)
        dec = hard_decision(bp_result, H)

        corr = η - lsd.basis * dec

        # lattice_statistics_decoding!(corr, lsd)
        if local_search
            local_search!(corr, y, dec, lsd)
        end

        res = y - corr

        # println(2 * logical * res)
        if (round.(Int, abs.(res * sqrt(2))).%2)[1] != 0
            1
        else
            0
        end
    end
    return tot_errors
end


@everywhere function qec_experiment(logical, H, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations, decoder, dec_style, search_radius)
    results = []
    tg = initialize_tanner_graph(H)
    lsd = LatticeStatisticsDecoding(order, G, reduced_basis)
    results = [qec_sample(tg, lsd, H, G, logical, σ, n_samples, local_search, schedule, iterations, decoder, dec_style, search_radius) for σ in sigmas]
    return results
end


global result_dict = Dict()
path = "results/rep_code_3.csv"
for decoder in ["lsd", "nearest"]
    for local_search in [true, false]
        for schedule in ["serial", "parallel"]
            for dec_style in ["syndrome", "received_vector"]
                for n in [3]
                    code = GKP_Rep_Code(n, false, true)
                    logical = Vector(code.logical[1:n])
                    order = local_search ? [2, 1, 1] : [0]
                    reduced_basis = true
                    search_radius = sqrt(1 / 2)
                    iterations = n


                    H = code.code[n+1:end, n+1:end]
                    G = inv(H)
                    sigmas = collect(0.3:0.1:0.4) ./ sqrt(2 * pi)
                    n_samples = 1_000
                    results = qec_experiment(logical, H, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations,
                        decoder, dec_style, search_radius)

                    for (σ, res) in zip(sigmas, results)
                        json_data = metadata(
                            code="rep_code",
                            schedule=schedule,
                            decoder=decoder,  # "lsd",
                            d=n,
                            local_seach_order=order,
                            local_search=local_search,
                            reduced_basis=reduced_basis,
                            sigma=σ,
                            iterations=iterations,
                            decoding_style=dec_style,
                        )
                        add_data!(path, shots=n_samples, errors=res, decoder=decoder, json_metadata=json_data)
                    end

                    # save results to file
                    # result_dict[n] = results
                end
            end
        end
    end
end

# using Plots
# p = plot(xlabel="σ", ylabel="P(failure)")
# sigmas = collect(0.3:0.1:1.0)
# n_samples = 150_000
# for (n, results) in result_dict
#     plot!(p, sigmas, results ./ n_samples, label="n = $n", marker=:circle, yscale=:log10, lw=2, msw=0.5, ms=2.5)
# end
# display(p)
