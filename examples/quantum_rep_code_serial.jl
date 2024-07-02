# using Distributed
# addprocs(6);
using LatticeDecoder

n = 3;
σ = 0.50;
n_samples = 1_000_000;

function calculate_error_rate(n, σ, n_samples)
    J = symplectic_form(n)
    code = GKP_Rep_Code(n, false, true)
    logical = vec(code.logical')
    M = code.code
    H = -M * J
    Mperp = -inv(J * M')
    G = inv(H)


    failures = 0
    tg = initialize_tanner_graph(H)
    for sample in 1:n_samples
        y = sample_error(σ, tg.nv)
        y[tg.nv÷2+1:end] .= 0.0
        tg.schedule = collect(tg.nv:-1:1)
        bp_result = run_serial_belief_propagation!(tg, y, σ, max(5, n), sqrt(1 / 2))
        dec = hard_decision(bp_result, H)
        res = G * dec
        success = !is_logical_error(code, res)
        if !success
            failures += 1
        end

        if failures > 1_000
            return failures / sample
        end
    end
    print("n: ", n, " sigma: ", σ, " failures: ", failures, " n_samples: ", n_samples, "\n")
    return failures / n_samples
end

using Plots
p = plot();
ns = [3, 5, 7, 9, 11];
sigmas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

COLORS = Dict(
    3 => :red,
    5 => :blue,
    7 => :green,
    9 => :purple,
    11 => :orange,
    13 => :brown
);

for n in ns
    error_rates = [calculate_error_rate(n, σ / sqrt(2pi), n_samples) for σ in sigmas]
    plot!(p, sigmas, error_rates, label="n = $n", yscale=:log10, marker=:square, color=COLORS[n])
end

matching_results = Dict(
    "5" => [0.0035, 0.0221, 0.067, 0.127, 0.1856, 0.2684, 0.3302, 0.3819, 0.4273, 0.4449, 0.4639, 0.4712, 0.4984, 0.5187, 0.4965, 0.5208],
    "13" => [0.0, 0.0005, 0.0084, 0.0343, 0.0927, 0.1548, 0.243, 0.3121, 0.3749, 0.4192, 0.4603, 0.4869, 0.5005, 0.5248, 0.55, 0.5547],
    "7" => [0.0012, 0.0098, 0.0341, 0.0914, 0.1564, 0.232, 0.2994, 0.3615, 0.4036, 0.4385, 0.4709, 0.476, 0.5083, 0.5072, 0.5122, 0.527],
    "11" => [0.0, 0.0012, 0.0145, 0.0465, 0.1082, 0.1826, 0.2576, 0.3282, 0.3857, 0.4251, 0.4642, 0.4867, 0.4955, 0.5096, 0.5292, 0.541],
    "9" => [0.0005, 0.0046, 0.0228, 0.0598, 0.1289, 0.2052, 0.2778, 0.3374, 0.3912, 0.43, 0.46, 0.4766, 0.5039, 0.5185, 0.527, 0.5333],
    "sigmas" => [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "3" => [0.0156, 0.0508, 0.1045, 0.1738, 0.2447, 0.3065, 0.3597, 0.394, 0.4266, 0.4591, 0.4815, 0.4888, 0.4953, 0.5047, 0.5048, 0.5065]
);

for n in ns
    if n < 11
        plot!(p, matching_results["sigmas"], matching_results[string(n)], label="n = $n (matching)", marker=:circle, linestyle=:dash, color=COLORS[n])
    end
end

# move legend to lower right
plot!(p, legend=:bottomright)
xlims!(p, 0.4, 1.0)


display(p)

