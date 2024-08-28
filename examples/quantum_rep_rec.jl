using Distributed
addprocs(5);
@everywhere using LatticeDecoder

# using LatticeDecoder

# code = GKP_Rep_Rec_Code(3)
# M = code.code
# L = code.logical

# Mc = 1. * code.code

# # multiply elemets of M randomly by (-1)
# function multiply_minus!(M::Matrix)
#     for idx in eachindex(M)
#         M[idx] *= rand([-1, 1])
#     end
# end

# multiply_minus!(M)
# M

# tg_m = initialize_tanner_graph(M);
# tg_p = initialize_tanner_graph(Mc);

# y = sample_error(0.5, tg.nv)

# bp_m = run_serial_belief_propagation!(tg_m, y, 0.3, 15)
# bp_p = run_serial_belief_propagation!(tg_p, y, 0.3, 15)

# println((bp_m - bp_p)')

# dec_m = hard_decision(bp_m, M)
# dec_p = hard_decision(bp_p, Mc)

# println((dec_m - dec_p)')

# res_m = inv(M) * dec_m
# res_p = inv(Mc) * dec_p

# L * res_p
# L * res_m

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
        if count(!=(0.0), (round.((logical*res)[1], digits=1) .% 1)) != 0
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
path = "results/rep_rec_code.csv"
for decoder in ["lsd", "nearest"]
    for local_search in [true]
        for schedule in ["serial", "parallel"]
            for dec_style in ["received_vector"]
                for n in 3:2:7
                    code = GKP_Rep_Rec_Code(n)
                    logical = code.logical
                    order = local_search ? [2, 1, 1] : [0]
                    reduced_basis = true
                    search_radius = 1 / sqrt(2)
                    iterations = n
                    H = code.code
                    G = inv(H)
                    sigmas = collect(0.3:0.1:1.0) ./ sqrt(2 * pi)
                    n_samples = 100_000
                    results = qec_experiment(logical, H, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations,
                        decoder, dec_style, search_radius)

                    for (σ, res) in zip(sigmas, results)
                        json_data = metadata(
                            code="rep_rec_code",
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
                end
            end
        end
    end
end
