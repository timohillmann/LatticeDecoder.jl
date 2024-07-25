using Distributed
addprocs(6);
@everywhere using LatticeDecoder
# using LatticeDecoder

@everywhere function random_bitstring(n)
    return rand(0:1, n)
end

@everywhere function random_bitstring!(b::Vector{Int64}, n)
    @inbounds for i = 1:n
        b[i] = rand(0:1)
    end
end


@everywhere function random_encoding_experiment(H, σ, max_iter, samples)
    errors = 0
    n = size(H, 1)
    b = zeros(Int64, n)
    G = generator_matrix(H)
    tg = initialize_tanner_graph(H)
    println("Running random encoding experiment with σ = ", σ, " with $(nworkers()) workers ")
    errors = @distributed (+) for i = 1:samples
        random_bitstring!(b, n)
        y = encode(b, G)
        y .+= sample_error(σ, n)
        bp_result = run_belief_propagation!(tg, y, σ, max_iter)
        # bp_result = run_serial_belief_propagation!(tg, y, σ, max_iter)
        dec = hard_decision(bp_result, H)
        count_symbol_errors(dec, b)
    end
    return errors / samples / size(H, 1)
end


@everywhere function ec_experiment(H, σ, max_iter, samples)
    tg = initialize_tanner_graph(H)
    println("Running experiment with σ = ", σ, " with $(nworkers()) workers ")
    errors = @distributed (+) for i = 1:samples
        y = sample_error(σ, size(H, 2))
        # bp_result = run_serial_belief_propagation!(tg, y, σ, max_iter)
        bp_result = run_belief_propagation!(tg, y, σ, max_iter)
        dec = hard_decision(bp_result, H)
        count_symbol_errors(dec)
    end

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

function classical_surface_code(d::Int64)
    code = GKP_Surface_Code(d)
    H = code.code
    n = d^2
    # cut out the perfect half
    H = H[n+1:end, n+1:end]
    return H
end


using Plots
samples = 25_000;
max_iter = 10;
σ = lattice_capacity_std() * sqrt(2);
p = plot();
sigmas = [0.25, 0.2, 0.15, 0.1]# range(σ, 0.3 * σ, 8);

d = 5;
for n in [5, 9, 13]
    H = classical_surface_code(n)
    # ber = [ec_experiment(H, σ, max_iter, samples) for σ in sigmas]

    ber = [random_encoding_experiment(H, σ, max_iter, samples) for σ in sigmas]
    ribbon = agresti_coull_confidence_interval.(ber, samples * n)
    println(ber)
    plot!(p, snr_db.(sigmas), ber, xlabel="σ (dB) from Capacity", ylabel="SER", label="[$(n^2),$(n)]", title="Symbol Error Rate vs. σ", lw=2,
        marker=:circle, markersize=5, grid=true, ribbon=ribbon)
end
# set x and y axis in log scale
plot!(p, yscale=:log10)
# display plot
display(p)