using Distributed
addprocs(18)
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
@everywhere using NPZ

println("Number of workers: ", nworkers())


function create_classical_ldlc(code_params::Dict)
    d = code_params["d"]
    n = code_params["n"]
    H = classical_ldlc(d, n, true)
    G = generator_matrix(H)
    code_name = "ldlc_n$(n)_d$(d)"

    # Save the matrices H and G to disk
    save_dir = "data/classical_ldlc_matrices"
    # Create directory if it doesn't exist
    if !isdir(save_dir)
        mkpath(save_dir)
    end
    npzwrite(joinpath(save_dir, "$(code_name)_H.npy"), H)
    npzwrite(joinpath(save_dir, "$(code_name)_G.npy"), G)
end


function load_classical_ldlc(code_params::Dict)
    d = code_params["d"]
    n = code_params["n"]
    code_name = "ldlc_n$(n)_d$(d)"
    load_dir = "data/classical_ldlc_matrices"
    H = npzread(joinpath(load_dir, "$(code_name)_H.npy"))
    G = npzread(joinpath(load_dir, "$(code_name)_G.npy"))
    return H, G, code_name
end

@everywhere function random_bitstring(n)
    return rand(0:1, n)
end

@everywhere function random_bitstring!(b::Vector{Int64}, n)
    @inbounds for i = 1:n
        b[i] = rand(0:1)
    end
end



@everywhere function random_encoding_experiment(
    tg::LatticeDecoder.TannerGraph,
    H::AbstractMatrix{Float64},
    G::AbstractMatrix{Float64},
    σ::Float64,
    n_samples;
    schedule::String = "serial",
    iterations::Int = tg.nv,
    decoder::String = "lsd",
    search_radius::Float64 = 1.5
)

    if schedule == "serial"
        run_bp! = run_serial_belief_propagation!
    else
        run_bp! = run_belief_propagation!
    end


    n = tg.nv
    b = zeros(Int64, n)
    errors = @distributed (+) for idx = 1:n_samples

        random_bitstring!(b, n)
        y = encode(b, G)
        y .+= sample_error(σ, tg.nv)

        bp_result = run_bp!(tg, y, σ, iterations, decoder, search_interval=search_radius)
        dec = hard_decision(bp_result, H)

        count_symbol_errors(dec, b)
    end

    return errors
end


@everywhere function run_experiment(H, G, sigmas, n_samples, schedule, iterations, decoder, search_radius)
    results = []
    tg = initialize_tanner_graph(H)
    results = [random_encoding_experiment(tg, H, G, σ, n_samples; schedule, iterations, decoder, search_radius) for σ in sigmas]
    return results
end


codes = [
    # Dict("n" => 1000, "d" => 7),
    Dict("n" => 10_000, "d" => 7)]

for code in codes
    create_classical_ldlc(code)
end


codes = [
    # Dict("n" => 128, "d" => 5),
    # Dict("n" => 256, "d" => 5),
    # Dict("n" => 512, "d" => 5),
    # Dict("n" => 1024, "d" => 5),
    # Dict("n" => 768, "d" => 7),
    # Dict("n" => 1024, "d" => 7)
    # Dict("n" => 1000, "d" => 7),
    Dict("n" => 10_000, "d" => 7)
    ]


global result_dict = Dict()
reduced_basis = true
path = "results/classical_codes/new_classical_ldlc.csv"
for r = 1:1
    for decoder in ["lsd"]
        for local_search in [false]
            for schedule in ["parallel"]
                for dec_style in ["received_vector"]
                    # for n in [3]
                    for code_params in codes
                        
                        H, G, code_name = load_classical_ldlc(code_params)
                        nbits = code_params["n"]
                        search_radius = 1.5
                        iterations = 30

                
                        σ = lattice_capacity_std()
                        sigmas = range(σ, .8 * σ, 6)[4:end];


                        n_samples = 100
                        results = run_experiment(
                            H, G, sigmas, n_samples, schedule, iterations, decoder, search_radius
                        )

                        for (σ, res) in zip(sigmas, results)
                            json_data = metadata(
                                code=code_name, 
                                schedule=schedule, # serial 
                                decoder=decoder,  # "lsd",
                                d=code_params["d"],
                                local_seach_order=0,
                                local_search=local_search,
                                reduced_basis=reduced_basis,
                                sigma=σ,
                                iterations=iterations,
                                decoding_style=dec_style,
                                nbits=nbits,
                            )
                            # we add shots = n_samples * nbits to plots the correct statistics, the symbol error rate.
                            add_data!(path, shots=n_samples * nbits, errors=res, decoder=decoder, json_metadata=json_data)
                        end

                        # save results to file
                        # result_dict[n] = results
                    end
                end
            end
        end
    end
end