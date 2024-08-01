using Distributed
addprocs(5);
@everywhere using LatticeDecoder

@everywhere function qec_sample(tg, lsd, H, G, logical, σ, n_samples, local_search::Bool=true, schedule::String="serial", iterations::Int=tg.nv)
    if schedule == "serial"
        run_bp! = run_serial_belief_propagation!
    else
        run_bp! = run_belief_propagation!
    end
    tot_errors = @distributed (+) for idx = 1:n_samples
        y = sample_error(σ, tg.nv)

        bp_result = run_bp!(tg, y, σ, iterations)
        # bp_result = run_serial_belief_propagation!(tg, y, σ, tg.nv)
        # bp_result = run_belief_propagation!(tg, y, σ, tg.nv)
        dec = hard_decision(bp_result, H)

        corr = y - lsd.basis * dec

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


@everywhere function qec_experiment(logical, H, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations)
    results = []
    tg = initialize_tanner_graph(H)
    lsd = LatticeStatisticsDecoding(order, G, reduced_basis)
    results = [qec_sample(tg, lsd, H, G, logical, σ, n_samples, local_search, schedule, iterations) for σ in sigmas]
    return results
end


global result_dict = Dict()
path = "results/rep_code_2.csv"
for local_search in [true, false]
    for schedule in ["serial", "parallel"]
        for n in 15:2:21
            code = GKP_Rep_Code(n, false, true)
            logical = Vector(code.logical[1:n])
            order = local_search ? [2, 1, 1] : [0]
            reduced_basis = true
            iterations = n


            H = code.code[n+1:end, n+1:end]
            G = inv(H)
            sigmas = collect(0.3:0.1:0.4) ./ sqrt(2 * pi)
            n_samples = 1_000_000
            results = qec_experiment(logical, H, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations)

            for (σ, res) in zip(sigmas, results)
                json_data = metadata(
                    code="rep_code",
                    schedule=schedule,
                    decoder="lsd",  # "lsd",
                    d=n,
                    local_seach_order=order,
                    local_search=local_search,
                    reduced_basis=reduced_basis,
                    sigma=σ,
                    iterations=iterations,
                )
                add_data!(path, shots=n_samples, errors=res, decoder="lsd", json_metadata=json_data)
            end

            # save results to file
            # result_dict[n] = results
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
