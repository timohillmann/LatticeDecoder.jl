using Distributed
addprocs(8);
@everywhere using LatticeDecoder

# Parameters
n = 256;
d = 5;
H = classical_ldlc(d, n, true);

# Initialize Tanner graph

tg = initialize_tanner_graph(H);

σ = 0.15;

y = sample_error(σ, n);

bp_result = run_belief_propagation!(tg, y, σ, 100);
dec = hard_decision(bp_result, H);

println("Number of symbol errors: ", count_symbol_errors(dec))



@everywhere function ec_experiment(H, σ, max_iter, samples)
    tg = initialize_tanner_graph(H)
    println("Running experiment with σ = ", σ, " with $(nworkers()) workers ")
    errors = @distributed (+) for i = 1:samples
        y = sample_error(σ, size(H, 2))
        bp_result = run_serial_belief_propagation!(tg, y, σ, max_iter)
        # bp_result = run_belief_propagation!(tg, y, σ, max_iter)
        dec = hard_decision(bp_result, H)
        count_symbol_errors(dec)
    end
    # errors = 0
    # for i = 1:samples
    #     y = sample_error(σ, size(H, 2))
    #     tg = initialize_tanner_graph(H)
    #     bp_result = run_belief_propagation!(tg, y, σ, max_iter)
    #     dec = hard_decision(bp_result, H)
    #     errors += count_symbol_errors(dec)
    # end

    return errors / samples / size(H, 1)
end


using Plots
samples = 1000;
max_iter = 10;
σ = lattice_capacity_std()
p = plot()
sigmas = range(σ, 0.85 * σ, 4)

d = 5
for n in [100,1000]
    ber = zeros(length(sigmas))
    for _ in 1:10
        # H = classical_ldlc(d, n, true)
        # H = load_ldlc(d, n)
        H = classical_ldlc(d, n, true)
        ber = ber + Vector([ec_experiment(H, σ, max_iter, samples) for σ in sigmas])
    end
    plot!(p, snr_db.(sigmas), ber ./10, xlabel="σ (dB) from Capacity", ylabel="SER", label="[$(n), $(d)]", title="Symbol Error Rate vs. σ", lw=1,
        marker=:circle, markersize=5, grid=true,color=:blue)
    # set x and y axis in log scale
    plot!(p, yscale=:log10)
    # display plot
end
display(p)