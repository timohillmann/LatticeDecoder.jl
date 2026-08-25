using Random
using Statistics
using Printf

using LatticeDecoder

function run_equivalence_case(H::AbstractMatrix, σ::Float64, iters::Int, seed::Int)
    n = size(H, 2)
    Random.seed!(seed)
    y = randn(n) .* σ

    tg_base = initialize_tanner_graph(H)
    tg_opt = initialize_tanner_graph(H)

    base = copy(LatticeDecoder.run_belief_propagation_lsd_reference!(tg_base, y, σ, iters))
    opt = copy(run_belief_propagation!(tg_opt, y, σ, iters, "lsd"))

    max_abs_diff = maximum(abs.(base .- opt))
    hd_base = hard_decision(base, H)
    hd_opt = hard_decision(opt, H)
    hard_decision_match = all(hd_base .== hd_opt)

    return max_abs_diff, hard_decision_match
end

function main()
    d = 5
    n = 384
    iters = 8
    sigmas = [0.15, 0.2, 0.3]
    seeds = [101, 202, 303, 404, 505]
    tol = 1e-8

    H = classical_ldlc(d, n)

    all_diffs = Float64[]
    all_hd_match = Bool[]

    println("LSD overlay equivalence check")
    println("n=$(n), d=$(d), iterations=$(iters)")

    for σ in sigmas
        for seed in seeds
            diff, hd_match = run_equivalence_case(H, σ, iters, seed)
            push!(all_diffs, diff)
            push!(all_hd_match, hd_match)
            @printf("sigma=%0.3f seed=%d  max_abs_diff=%0.3e  hard_decision_match=%s\n", σ, seed, diff, string(hd_match))
        end
    end

    max_diff = maximum(all_diffs)
    mean_diff = mean(all_diffs)
    all_match = all(all_hd_match)

    println("--- Summary ---")
    @printf("max_abs_diff=%0.3e\n", max_diff)
    @printf("mean_abs_diff=%0.3e\n", mean_diff)
    println("hard_decision_all_match=$(all_match)")

    if max_diff > tol || !all_match
        error("Equivalence check failed: max_diff=$(max_diff), hard_decision_all_match=$(all_match)")
    end

    println("Equivalence check PASSED.")
end

main()
