using Distributed
addprocs(4)
@everywhere using LatticeDecoder
using PyPlot
using Plots

mutable struct Results
    ns::Vector{Int}
    σs::Vector{Float64}
    samples::Int
    max_iter::Int
    results::Dict{Int,Vector{Float64}}

    function Results(ns::Vector{Int}, σs::Vector{Float64}, samples::Int, max_iter::Int)
        results = Dict{Int,Vector{Float64}}()
        for n in ns
            results[n] = zeros(Float64, length(σs))
        end
        return new(ns, σs, samples, max_iter, results)
    end

    Results(ns::Vector{Int}, σs::StepRangeLen, samples::Int, max_iter::Int) = Results(ns, collect(σs), samples, max_iter)

end


σs = range(0.5, 2.0, 6)
ns = [3, 5, 7]
samples = 10
max_iter = 5

results = Results(ns, σs, samples, max_iter)

@everywhere function main(ns, σs, samples, max_iter, results)
    for n in ns
        println("n = ", n)
        J = symplectic_form(n)
        code = GKP_Rep_Code(n, false, true)
        logical = vec(code.logical')
        M = code.code
        H = -M * J
        println(H)
        # Mperp = -inv(J * M')

        G = inv(H)

        tg = initialize_tanner_graph(H)
        logical[1] = 1
        logical = zeros(Int8, 1, tg.nv)
        for (σ_id, σ) in enumerate(σs)
            σ = σ / sqrt(2 * pi)

            success = @distributed (+) for i = 1:samples
                y = sample_error(σ, tg.nv)
                y[tg.nv÷2+1:end] .= 0.0
                bp_result = run_belief_propagation!(tg, y, σ, max_iter)
                dec = hard_decision(bp_result, H)
                res_err = G * dec .% 2
                # println(res_err)
                if res_err[1] == 0
                    return 1
                else
                    return 0

                end
            end
            results.results[n][σ_id] = success / samples
        end
    end
    return results
end


results = main(ns, σs, samples, max_iter, results)

function plot_results(results::Results)
    fig = Plots.plot()
    for n in results.ns
        plot!(fig, results.σs, 1.0 .- results.results[n], label="n = $n")
    end
    plot!(fig, xlabel=L"$\sigma$", ylabel="Success rate", legend=:bottomright, yscale=:log)
    fig
end


plot_results(results)