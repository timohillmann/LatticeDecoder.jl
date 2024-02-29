# using LatticeDecoder
using Base.Threads
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/parallel_bp.jl");
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/tanner_graph.jl");
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/code_constructors/classical_ldlc.jl");

# Parameters
n = 256;
d = 5;
H = classical_ldlc(d, n, true);

# Initialize Tanner graph

tg = initialize_tanner_graph(H);

σ = 0.15;

"""
    sample_error(σ::Float64, n::Int)

Sample a Gaussian error vector with standard deviation `σ` and length `n`.
"""
function sample_error(σ::Float64, n::Int)
    return σ * randn(n)
end


"""
    hard_decision(bp_result::Vector{Float64}, H::AbstractArray)

Compute the hard decision using the belief propagation result.
"""
function hard_decision(bp_result::Vector{Float64}, H::AbstractArray)
    return Int64.(round.(H * bp_result))
end


"""
    count_bit_errors(x::AbstractArray)

Count the number of bit errors in the decoded codeword. The codeword is assumed to be the all-zero vector.
"""
function count_symbol_errors(x::AbstractArray)
    return sum(x .!= 0)
end





y = sample_error(σ, n);

bp_result = run_belief_propagation!(tg, y, σ, 100);
dec = hard_decision(bp_result, H);

println("Number of bit errors: ", count_symbol_errors(dec))



"""
    ec_experiment(H, σ, max_iter, samples)

Run an error correction experiment on a parity-check matrix `H` with a Gaussian error vector of standard deviation `σ`.
The experiment is repeated `samples` times and the average number of bit errors, i.e., the bit error rate, is returned.
"""
function ec_experiment(H, σ, max_iter, samples)
    tg = initialize_tanner_graph(H)
    errors = 0
    for i = 1:samples
        y = sample_error(σ, size(H, 2))
        tg = initialize_tanner_graph(H)
        bp_result = run_belief_propagation!(tg, y, σ, max_iter)
        dec = hard_decision(bp_result, H)
        errors += count_symbol_errors(dec)
    end

    return errors / samples / size(H, 1)
end

function lattice_capacity_std()
    return 1 / sqrt(2 * pi * ℯ)
end

function signal_to_noise_ratio(σ::Float64)
    return 1 / (2 * pi * exp(1) * σ^2)
end

snr(σ::Float64) = signal_to_noise_ratio(σ)

function snr_db(σ::Float64)
    return 10 * log10(snr(σ))
end


using Plots
samples = 5000;
max_iter = 15;
σ = lattice_capacity_std()
p = plot()
sigmas = range(σ, 0.8 * σ, 4)

d = 5
for n in [100, 256]
    H = classical_ldlc(d, n, true)
    ber = [ec_experiment(H, σ, max_iter, samples) for σ in sigmas]
    plot!(p, snr_db.(sigmas), ber, xlabel="σ (dB) from Capacity", ylabel="SER", label="[$(n), $(d)]", title="Symbol Error Rate vs. σ", lw=2,
        marker=:circle, markersize=5, legend=:topleft, grid=true)
end
# set x and y axis in log scale
plot!(p, yscale=:log10)
# display plot
display(p)