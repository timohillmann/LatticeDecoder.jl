using LatticeDecoder
using Plots

ns = [5, 9, 13]
σs = [0.1, 0.15, 0.2, 0.25] #  0.2:0.05:0.4
n_samples = 100_000
p = plot(xlabel="σ", ylabel="P(failure)")

for n in ns
    code = GKP_Rep_Code(n, false, true)
    logical = code.logical
    M = code.code
    H = -M * code.J

    # since the other half is perfect, decoding can be simplied.
    H = H[n+1:end, 1:n]
    G = inv(H)
    logical = logical[1:n]
    tg = initialize_tanner_graph(H)
    results = []
    for σ in σs
        printstyled("Running σ = $(σ)\n", color=:red, bold=true)
        failures = 0
        for s = 1:n_samples
            y = sample_error(σ, tg.nv)
            bp_result = run_serial_belief_propagation!(tg, y, σ, n)
            dec = hard_decision(bp_result, H)
            res = G * dec
            failures += abs(logical' * res) % 1 ≈ 0.0 ? 0 : 1
        end
        push!(results, failures / n_samples)
    end

    plot!(p, σs, results, label="n = $n", yscale=:log10, marker=:circle, lw=2, msw=0.5, ms=2.5)
end
display(p)
