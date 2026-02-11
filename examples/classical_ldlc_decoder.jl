using Distributed
addprocs(17);
@everywhere using LatticeDecoder
using Plots

@everywhere function random_encoding_experiment(H, σ::Float64, max_iter::Int64, samples::Int64, M::Int64)
    errors = 0
    n = size(H, 1)
    b = zeros(Int64, n)
    G = generator_matrix(H)
    tg = initialize_tanner_graph(H)
    decoder = LDLCDecoder(tg, M)
    # println("Running random encoding experiment with σ = ", σ, " with $(nworkers()) workers ")
    errors = @distributed (+) for i = 1:samples
        random_bitstring!(b, n)
        y = encode(b, G)
        y .+= sample_error(σ, n)
        bp_result = run_decoder!(decoder, y, σ, max_iter)
        dec = hard_decision(bp_result, H)
        count_symbol_errors(dec, b)
    end
    return (errors + 5) / samples / size(H, 1)
end

# function main(;samples::Int64 = 5000, max_iter::Int64=25, M::Int64=2, d::Int64=5, ns::AbstractVector{Int64}=[128, 256, 512])    

samples = 100
max_iter = 25
M = 2
d = 5
ns = [128, 256, 512]

σ = lattice_capacity_std();
sigmas = range(σ, 0.75 * σ, 6);

p = plot();
for n in ns
    H = classical_ldlc(d, n, true)
    println("Running n=$n and degree d=$d")
    ber = [random_encoding_experiment(H, σ, max_iter, samples, M) for σ in sigmas]
    ribbon = agresti_coull_confidence_interval.(ber, samples * n)
    plot!(p, snr_db.(sigmas), ber, xlabel="σ (dB) from Capacity", ylabel="SER", label="[$(n), $(d)]", title="Symbol Error Rate vs. σ", lw=2,
        marker=:circle, markersize=5, grid=true)
end
# end
plot!(p, yscale=:log10)
display(p)