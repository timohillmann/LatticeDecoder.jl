using Distributed
addprocs(5);
@everywhere using LatticeDecoder


@everywhere function qec_sample(tg, H, G, σ, n_samples, schedule::String="serial", iterations::Int=tg.nv, decoder::String="lsd", search_radius::Float64=1.5)
    if schedule == "serial"
        run_bp! = run_serial_belief_propagation!
    else
        run_bp! = run_belief_propagation!
    end
    b = zeros(Int64, tg.nv)
    tot_errors = @distributed (+) for idx = 1:n_samples
        random_bitstring!(b, n)
        y = encode(b, G)
        y .+= sample_error(σ, n)

        bp_result = run_bp!(tg, y, σ, iterations, decoder, search_interval=search_radius)
        dec = hard_decision(bp_result, H)

        count_symbol_errors(dec, b)
    end
    return tot_errors
end


@everywhere function qec_experiment(H, G, sigmas, n_samples, schedule, iterations, decoder, search_radius)
    results = []
    tg = initialize_tanner_graph(H)
    results = [qec_sample(tg, H, G, σ, n_samples, schedule, iterations, decoder, search_radius) for σ in sigmas]
    return results
end


global result_dict = Dict()
path = "results/rep_code_new.csv"
for decoder in ["lsd", "nearest"]
    for local_search in [true, false]
        for schedule in ["serial", "parallel"]
            for dec_style in ["syndrome", "received_vector"]
                for n in 3:2:17
                    code = load_ldlc(n, d)
                    iterations = n


                    H = code.code[n+1:end, n+1:end]
                    G = inv(H)
                    sigmas = collect(0.3:0.1:1.0) ./ sqrt(2 * pi)
                    n_samples = 100_000
                    results = qec_experiment(H, G, sigmas, n_samples, schedule, iterations,
                        decoder, search_radius)

                    for (σ, res) in zip(sigmas, results)
                        json_data = metadata(
                            code="$(n)_$(d)",
                            schedule=schedule,
                            decoder=decoder,  # "lsd",
                            degree=d,
                            size=n,
                            sigma=σ,
                            iterations=iterations,
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
