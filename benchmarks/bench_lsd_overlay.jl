using Random
using Statistics
using Printf

using LatticeDecoder

include(joinpath(@__DIR__, "..", "src", "perf_overrides", "LSDPerfOverlay.jl"))
using .LSDPerfOverlay

function bench_run(f::Function, label::AbstractString; reps::Int=6)
    times = Float64[]
    allocs = Int[]

    for _ in 1:reps
        push!(allocs, @allocated f())
        t = @elapsed f()
        push!(times, t)
    end

    return (
        label=label,
        mean_time=mean(times),
        min_time=minimum(times),
        max_time=maximum(times),
        mean_alloc=mean(allocs),
    )
end

function main()
    Random.seed!(42)

    d = 5
    n = 384
    σ = 0.2
    iters = 8
    reps = 8

    H = classical_ldlc(d, n)
    tg_base = initialize_tanner_graph(H)
    tg_opt = initialize_tanner_graph(H)

    # Warmup
    y = randn(n) .* σ
    run_belief_propagation!(tg_base, y, σ, iters, "lsd")
    run_belief_propagation_lsd_optimized!(tg_opt, y, σ, iters)

    baseline = bench_run("baseline_lsd"; reps=reps) do
        y_local = randn(n) .* σ
        run_belief_propagation!(tg_base, y_local, σ, iters, "lsd")
    end

    optimized = bench_run("overlay_lsd_optimized"; reps=reps) do
        y_local = randn(n) .* σ
        run_belief_propagation_lsd_optimized!(tg_opt, y_local, σ, iters)
    end

    speedup = baseline.mean_time / optimized.mean_time
    alloc_reduction = 1.0 - (optimized.mean_alloc / baseline.mean_alloc)

    println("LSD Overlay Benchmark")
    println("n=$(n), d=$(d), sigma=$(σ), iterations=$(iters), reps=$(reps)")
    println("--------------------------------------------------------------")
    @printf("%-24s  mean=%8.5f s  min=%8.5f s  max=%8.5f s  alloc=%12.1f B\n",
        baseline.label, baseline.mean_time, baseline.min_time, baseline.max_time, baseline.mean_alloc)
    @printf("%-24s  mean=%8.5f s  min=%8.5f s  max=%8.5f s  alloc=%12.1f B\n",
        optimized.label, optimized.mean_time, optimized.min_time, optimized.max_time, optimized.mean_alloc)
    println("--------------------------------------------------------------")
    @printf("speedup:          %0.3fx\n", speedup)
    @printf("alloc reduction:  %0.2f%%\n", 100 * alloc_reduction)

    meets_speed = speedup >= 2.0
    meets_alloc = alloc_reduction >= 0.80

    println("acceptance_speedup>=2x:           $(meets_speed)")
    println("acceptance_alloc_reduction>=80%:  $(meets_alloc)")
end

main()
