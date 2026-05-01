using Random
using Statistics
using Printf

using LatticeDecoder

try
    using BenchmarkTools
catch
    error("BenchmarkTools.jl is required. Install with: julia --project=. -e 'using Pkg; Pkg.add(\"BenchmarkTools\")'")
end

include(joinpath(@__DIR__, "..", "src", "perf_overrides", "LSDPerfOverlay.jl"))
using .LSDPerfOverlay

function main(; samples::Int=200)
    Random.seed!(42)

    d = 7
    n = 984
    σ = 0.22
    iters = 25

    H = classical_ldlc(d, n)

    println("BenchmarkTools LSD benchmark")
    println("n=$(n), d=$(d), sigma=$(σ), iterations=$(iters), samples=$(samples)")

    baseline_trial = @benchmark run_belief_propagation!(tg, y, $σ, $iters, "lsd") setup=(tg=initialize_tanner_graph($H); y=randn($n) .* $σ) evals=1 samples=samples

    overlay_trial = @benchmark run_belief_propagation_lsd_optimized!(tg, y, $σ, $iters) setup=(tg=initialize_tanner_graph($H); y=randn($n) .* $σ) evals=1 samples=samples

    println("\n--- Baseline Trial ---")
    show(stdout, MIME"text/plain"(), baseline_trial)
    println()

    println("\n--- Overlay Trial ---")
    show(stdout, MIME"text/plain"(), overlay_trial)
    println()

    base_med_ns = median(baseline_trial).time
    over_med_ns = median(overlay_trial).time
    speedup = base_med_ns / over_med_ns

    println("\n--- Summary ---")
    @printf("baseline median: %.3f ms\n", base_med_ns / 1e6)
    @printf("overlay  median: %.3f ms\n", over_med_ns / 1e6)
    @printf("median speedup:  %.3fx\n", speedup)
end

# Optional CLI: julia --project=. benchmarks/bench_lsd_overlay_benchmarktools.jl 30
samples = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
main(samples=samples)
