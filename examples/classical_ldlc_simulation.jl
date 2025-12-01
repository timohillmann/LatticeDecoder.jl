using Distributed
addprocs(6);
@everywhere using LatticeDecoder
# using Revise
# using LatticeDecoder

# Parameters
n = 256;
d = 5;
H = classical_ldlc(d, n, true);

# Initialize Tanner graph

tg = initialize_tanner_graph(H);

σ = 0.20;

b = zeros(Int64, size(H, 1));
b[1] = 0;
b[2] = 0;

G = generator_matrix(H);
y = encode(b, G);

y .+= sample_error(σ, n);

# lp_result = l1_minimize(y, H)
# dec_lp = hard_decision(lp_result, H)

serial_bp_result = run_belief_propagation!(tg, y, σ, 30, "nearest");
serial_dec = hard_decision(serial_bp_result, H);

bp_result = run_belief_propagation!(tg, y, σ, 20, 3);
dec = hard_decision(bp_result, H);



println("Number of symbol errors: ", count_symbol_errors(dec, b))
println("Number of symbol errors (serial): ", count_symbol_errors(serial_dec, b))
# println("Number of symbol errors (LP): ", count_symbol_errors(dec_lp, b))

# check that bp_results approximately fulfill the parity check equations
println(round.(Int64, H * bp_result) .% 1)


@everywhere function random_bitstring(n)
    return rand(0:1, n)
end

@everywhere function random_bitstring!(b::Vector{Int64}, n)
    @inbounds for i = 1:n
        b[i] = rand(0:1)
    end
end


@everywhere function random_encoding_experiment(H, σ, max_iter, samples, decoder)
    errors = 0
    n = size(H, 1)
    b = zeros(Int64, n)
    G = generator_matrix(H)
    tg = initialize_tanner_graph(H)
    println("Running random encoding experiment with σ = ", σ, " with $(nworkers()) workers ")
    errors = @distributed (+) for i = 1:samples
        # random_bitstring!(b, n)
        y = encode(b, G)
        y .+= sample_error(σ, n)
        bp_result = run_belief_propagation!(tg, y, σ, max_iter, decoder)
        # bp_result = run_serial_belief_propagation!(tg, y, σ, max_iter, decoder)
        dec = hard_decision(bp_result, H)

        # lp_result = l1_minimize(y, H)
        # dec = hard_decision(lp_result, H)

        count_symbol_errors(dec, b)
    end
    return errors / samples / size(H, 1)
end


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


"""
    agresti_coull_confidence_interval(p, n, z=1.96)

Compute the Agresti-Coull confidence interval for a binomial distribution.
The default value of z is for a 95% confidence interval.
To get the 99% confidence interval, use z=2.576.
To get the 99.9% confidence interval, use z=3.291.

"""
function agresti_coull_confidence_interval(p, n, z=1.96)
    p̂ = p
    n̂ = n
    z = z
    p̂_ = (n̂ * p̂ + z^2 / 2) / (n̂ + z^2)
    n̂_ = n̂ + z^2
    return z * sqrt(p̂_ * (1 - p̂_) / n̂_)
end

using Plots
samples = 1000;
max_iter = 10;
σ = lattice_capacity_std()
p = plot()
sigmas = range(σ, 0.75 * σ, 5)
samples = 250;
max_iter = 50;
σ = lattice_capacity_std();
p = plot();
sigmas = range(σ, 0.75 * σ, 6);

decoder = 3;
d = 5;
for n in [128, 256, 512]
    H = classical_ldlc(d, n, true)
    println("Got code.")
    # ber = [ec_experiment(H, σ, max_iter, samples) for σ in sigmas]

    ber = [random_encoding_experiment(H, σ, max_iter, samples, decoder) for σ in sigmas]
    ribbon = agresti_coull_confidence_interval.(ber, samples * n)
    println(ber)
    plot!(p, snr_db.(sigmas), ber, xlabel="σ (dB) from Capacity", ylabel="SER", label="[$(n), $(d)]", title="Symbol Error Rate vs. σ", lw=2,
        marker=:circle, markersize=5, grid=true, ribbon=ribbon)
end
# set x and y axis in log scale
plot!(p, yscale=:log10)
# display plot
display(p)