using Distributed
@everywhere using NormalForms
@everywhere using LLLplus
@everywhere using LatticeDecoder
using DelimitedFiles
using NPZ
using LinearAlgebra

function load_txt_matrix(file::String)
    return readdlm(file, String)
end

function convert_to_vec(vec::SubString)
    return [parse(Int, x) for x in split(vec, "")]
end

function load_erik_code()
    path = "/Users/timo/Documents/GitHub/LatticeDecoder.jl/nithin_codes_N_544_K_80_L_16_hx_lattice.txt"
    hx = load_txt_matrix(path)
    new_hx = []
    for i in 1:size(hx, 1)
        # println(i, " ", hx[i, 1])
        push!(new_hx, [parse(Int, x) for x in split(hx[i, 1], "")])
    end
    new_hx = hcat(new_hx...)

    # load logical
    logical = npzread("/Users/timo/Documents/GitHub/LatticeDecoder.jl/data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_lz.npy")
    Gx = npzread("/Users/timo/Documents/GitHub/LatticeDecoder.jl/data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_gx.npy")
    return new_hx / sqrt(2), logical / sqrt(2), Gx / sqrt(2)
end

"""
    gkp_rep_code(d::Int, bit_flip=false)

Construct the GKP-Repetition code with `d` modes. If `bit_flip` is true, then the protected qubit is encoded in the position basis, otherwise it is encoded in the momentum basis.
"""
function nithin_code()

    H = npzread("/Users/timo/Documents/GitHub/LatticeDecoder.jl/data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_hx.npz")
    logical = npzread("/Users/timo/Documents/GitHub/LatticeDecoder.jl/data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_lz.npy")
    n = 544
    GKP_generators = Matrix{Float64}(2 * I, n, n)

    return [GKP_generators; H] / sqrt(2), logical / sqrt(2)
end


@everywhere function qec_sample(tg, lsd, H, H_R, G, logical, σ, n_samples, local_search::Bool=true, schedule::String="serial", iterations::Int=tg.nv, decoder::String="lsd",
    decoding_style::String="syndrome", search_radius::Float64=1.5)

    # H_R = inv(G)
    # H_R, _ = lll(H_R)

    printstyled("Starting sample\n", color=:blue)

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
        dec = hard_decision(bp_result, H_R)

        corr = η - lsd.basis * dec

        # lattice_statistics_decoding!(corr, lsd)
        if local_search
            local_search!(corr, y, dec, lsd)
        end

        res = y - corr

        # println(2 * logical * res)
        # check if any of `(round.(Int, abs.(logical * res)).%2)` is non-zero\
        printstyled("Log Com. Check: $((round.(Int, abs.(logical * res))))", color=:green)
        log_check = (round.(Int, abs.(logical * res)) .% 2) .!= 0
        println(log_check)
        if any(log_check)
            1
        else
            0
        end
    end
    return tot_errors
end


@everywhere function qec_experiment(logical, H, H_R, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations, decoder, dec_style, search_radius)
    results = []
    printstyled("initialize tanner graph\n", color=:blue)
    tg = initialize_tanner_graph(H)
    printstyled("initialize lsd\n", color=:blue)
    lsd = LatticeStatisticsDecoding(order, G, reduced_basis)
    printstyled("start experiment\n", color=:red)
    results = [qec_sample(tg, lsd, H, H_R, G, logical, σ, n_samples, local_search, schedule, iterations, decoder, dec_style, search_radius) for σ in sigmas]
    return results
end


global result_dict = Dict()
reduced_decoding = false
path = "results/nithin_code_160924_$(reduced_decoding)_test.csv"
for decoder in ["lsd"]
    for local_search in [false]
        for schedule in ["parallel"]
            for dec_style in ["received_vector"]
                for n in [30]
                    H, logical = nithin_code()
                    H_R, L, G = load_erik_code()
                    # logical = Vector(code.logical[1:n])'
                    # println("logical: ", logical)
                    order = local_search ? [2, 1, 1] : [0]
                    reduced_basis = false
                    search_radius = sqrt(1 / 2)
                    iterations = 1000



                    sigmas = [0.001]  # collect(0.03:0.01:0.1) ./ sqrt(2 * pi)
                    n_samples = 5
                    println("Starting experiment")
                    results = qec_experiment(logical, H, H_R, G, sigmas, n_samples, order, local_search, schedule, reduced_basis, iterations,
                        decoder, dec_style, search_radius)

                    for (σ, res) in zip(sigmas, results)
                        json_data = metadata(
                            code="nithin_code",
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


