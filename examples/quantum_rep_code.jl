using Distributed
addprocs(5);
@everywhere using LatticeDecoder

@everywhere function qec_sample(tg, lsd, H, G, logical, σ, n_samples)
    tot_errors = @distributed (+) for idx = 1:n_samples
        y = sample_error(σ, tg.nv)


        bp_result = run_serial_belief_propagation!(tg, y, σ, tg.nv)
        dec = hard_decision(bp_result, H)

        corr = y - lsd.basis * dec

        # lattice_statistics_decoding!(corr, lsd)
        local_search!(corr, y, dec, lsd)

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


@everywhere function qec_experiment(logical, H, G, sigmas, n_samples)
    results = []


    order = [2, 1, 1]
    tg = initialize_tanner_graph(H)
    lsd = LatticeStatisticsDecoding(order, G, false)

    results = [qec_sample(tg, lsd, H, G, logical, σ, n_samples) for σ in sigmas]
    return results
end


global result_dict = Dict()
for n in [3, 5, 7, 9]
    code = GKP_Rep_Code(n, false, true)
    logical = Vector(code.logical[1:n])
    # println("Logical: ", logical)
    H = code.code[n+1:end, n+1:end]
    G = inv(H)
    sigmas = collect(0.5:0.1:1.0) ./ sqrt(2 * pi)
    n_samples = 50_000
    results = qec_experiment(logical, H, G, sigmas, n_samples)
    # save results to file
    result_dict[n] = results
end

using Plots
p = plot(xlabel="σ", ylabel="P(failure)")
sigmas = collect(0.5:0.1:1.0)
n_samples = 50_000
for (n, results) in result_dict
    plot!(p, sigmas, results ./ n_samples, label="n = $n", marker=:circle, yscale=:log10, lw=2, msw=0.5, ms=2.5)
end
display(p)