using Distributed
addprocs(7);
@everywhere using LatticeDecoder


@everywhere function random_bitstring(n)
    return rand(0:1, n)
end

@everywhere function random_bitstring!(b::Vector{Int64}, n)
    @inbounds for i = 1:n
        b[i] = rand(0:1)
    end
end


@everywhere function random_encoding_experiment(H, σ, max_iter, samples, decoder, schedule)
    errors = 0
    n = size(H, 1)
    b = zeros(Int64, n)
    G = generator_matrix(H)
    tg = initialize_tanner_graph(H)

    if schedule == "serial"
        run_bp! = run_serial_belief_propagation!
    else
        run_bp! = run_belief_propagation!
    end

    println("Running random encoding experiment with σ = ", σ, " with $(nworkers()) workers ")
    errors = @distributed (+) for i = 1:samples
        random_bitstring!(b, n)
        y = encode(b, G)
        y .+= sample_error(σ, n)
        bp_result = run_bp!(tg, y, σ, max_iter, decoder)
        # bp_result = run_serial_belief_propagation!(tg, y, σ, max_iter, decoder)
        dec = hard_decision(bp_result, H)

        count_symbol_errors(dec, b)
    end
    return errors
end



path = "results/classical_ldlc/results.csv"
σ = lattice_capacity_std()
n_samples = 10_000
max_iter = 50
sigmas = range(σ, 0.75 * σ, length=10)
for n in [128, 256, 512, 1024]
    for d in [5]
        H = classical_ldlc(d, n, true)
        for schedule in ["serial", "parallel"]
            for decoder in ["nearest", "lsd"]
                results = [random_encoding_experiment(H, σ, max_iter, n_samples, decoder, schedule) for σ in sigmas]

                for (σ, res) in zip(sigmas, results)
                    json_data = metadata(
                        code="cldlc_$(n)_$(d)",
                        schedule=schedule,
                        decoder=decoder,  # "lsd",
                        sigma=σ,
                        iterations=max_iter,
                        nbits=size(H, 2),
                    )
                    println(res)
                    # we add shots = n_samples * nbits to plots the correct statistics, the symbol error rate.
                    add_data!(path, shots=n_samples * size(H, 2), errors=res, decoder=decoder, json_metadata=json_data)
                end

                # save results to file
                # result_dict[n] = results
            end
        end
    end
end