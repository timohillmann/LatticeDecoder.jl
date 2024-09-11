using Distributed
if nprocs()<8
    addprocs(8-nprocs())
end
@everywhere using LatticeDecoder
using PyPlot
using Plots
using Optim


# σs = range(0.5, 2.0, 6)

n = 3

samples = 10_000
max_iter = 20

@everywhere function cost(multipliers::Vector{Float64}, pars)
    tg, H, G, σs, samples, max_iter = pars
   
    σ = σs

    success = @distributed (+) for i = 1:samples
        y = sample_error(σ, tg.nv)
        # bp_result = run_belief_propagation!(tg, y, σ*multipliers, max_iter, "nearest", search_interval=1/sqrt(2))
        bp_result = run_belief_propagation!(tg, y, σ*multipliers, max_iter, "lsd")

        dec = hard_decision(bp_result, H)
        res_err = G * dec
        # println(res_err)
        if (round.(Int, abs.(res_err * sqrt(2))).%2)[1] != 0
            0
        else
            1
        end
    end
    # println(success/samples)
    return -success / samples
end


@everywhere function just_run(n,σ, samples, max_iter)
        # println("n = ", n)
        J = symplectic_form(n)
        code = GKP_Rep_Code(n, false, true)
        # logical = vec(code.logical')
        H = code.code[n+1:end,n+1:end]
    
        G = inv(H)
    
        tg = initialize_tanner_graph(H)

   
        success = @distributed (+) for i = 1:samples
            y = sample_error(σ, tg.nv)
            # bp_result = run_belief_propagation!(tg, y, σ, max_iter, "nearest", search_interval=1/sqrt(2))
            bp_result = run_belief_propagation!(tg, y, σ, max_iter, "lsd")
            dec = hard_decision(bp_result, H)
            res_err = G * dec
            # println(res_err)
            if (round.(Int, abs.(res_err * sqrt(2))).%2)[1] != 0
                0
            else
                1
            end
        end
    # println(success/samples)
    return -success / samples
end

function optimize_rep_code(n,σ, samples, max_iter; time_limit=NaN)
    # println("n = ", n)
    J = symplectic_form(n)
    code = GKP_Rep_Code(n, false, true)
    # logical = vec(code.logical')
    H = code.code[n+1:end,n+1:end]

    G = inv(H)

    tg = initialize_tanner_graph(H)
    #  logical[1] = 1
    #  logical = zeros(Int8, 1, tg.nv)
    pars = (tg, H, G, σ, samples, max_iter)
    # starting_multipliers = ones(Float64, n)
    # starting_multipliers = rand(n) .+ 0.5
    starting_multipliers = 5*rand(n)
    println(starting_multipliers)
    return starting_multipliers, Optim.optimize(x->(cost(x, pars)), starting_multipliers, method=NelderMead(), show_trace=true, allow_f_increases=true; time_limit = time_limit) #time_limit=60.0
    # return starting_multipliers, optimize(x->(cost(x, pars)), zeros(Float64, n), 5*ones(Float64, n), starting_multipliers, Fminbox(NelderMead()), Optim.Options(show_trace=true)) #time_limit=60.0    # 
    
end

using Evolutionary
function optimize_rep_code_evo(n,σ, samples, max_iter; time_limit=NaN)
    # println("n = ", n)
    J = symplectic_form(n)
    code = GKP_Rep_Code(n, false, true)
    # logical = vec(code.logical')
    H = code.code[n+1:end,n+1:end]

    G = inv(H)

    tg = initialize_tanner_graph(H)
    #  logical[1] = 1
    #  logical = zeros(Int8, 1, tg.nv)
    pars = (tg, H, G, σ, samples, max_iter)
    # starting_multipliers = ones(Float64, n)
    # starting_multipliers = rand(n) .+ 0.5
    starting_multipliers = 10*rand(n)
    println(starting_multipliers)
    return starting_multipliers, Evolutionary.optimize(x->(cost(x, pars)), starting_multipliers, CMAES(), Evolutionary.Options(iterations=100, show_trace=true, show_every=1, time_limit=Float64(time_limit))) 
end


####### OPTIM OPTIMIZATION
# starting_multipliers, results = optimize_rep_code(3, 0.5, samples,max_iter, time_limit = 30)
# starting_multipliers'
# Optim.minimizer(results)'
# results.minimum


###### EVOLUTIONARY OPTIMIZATION
# starting_multipliers, results = optimize_rep_code_evo(3, 0.5, samples,max_iter)
# results.minimum
# results.minimizer



################################ THRESHOLD Plots


matching_results = Dict(
    "5" => [0.0035, 0.0221, 0.067, 0.127, 0.1856, 0.2684, 0.3302, 0.3819, 0.4273, 0.4449, 0.4639, 0.4712, 0.4984, 0.5187, 0.4965, 0.5208],
    "13" => [0.0, 0.0005, 0.0084, 0.0343, 0.0927, 0.1548, 0.243, 0.3121, 0.3749, 0.4192, 0.4603, 0.4869, 0.5005, 0.5248, 0.55, 0.5547],
    "7" => [0.0012, 0.0098, 0.0341, 0.0914, 0.1564, 0.232, 0.2994, 0.3615, 0.4036, 0.4385, 0.4709, 0.476, 0.5083, 0.5072, 0.5122, 0.527],
    "11" => [0.0, 0.0012, 0.0145, 0.0465, 0.1082, 0.1826, 0.2576, 0.3282, 0.3857, 0.4251, 0.4642, 0.4867, 0.4955, 0.5096, 0.5292, 0.541],
    "9" => [0.0005, 0.0046, 0.0228, 0.0598, 0.1289, 0.2052, 0.2778, 0.3374, 0.3912, 0.43, 0.46, 0.4766, 0.5039, 0.5185, 0.527, 0.5333],
    "sigmas" => [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "3" => [0.0156, 0.0508, 0.1045, 0.1738, 0.2447, 0.3065, 0.3597, 0.394, 0.4266, 0.4591, 0.4815, 0.4888, 0.4953, 0.5047, 0.5048, 0.5065]
)





# σs = collect(0.6:0.3:2.1)
σs = collect(0.5:0.2:2) ./sqrt(2*pi)

distances = collect(3:2:13)

failure_probs = Array{Float64}(undef, length(distances), length(σs))
failure_probs_no_opt = Array{Float64}(undef, length(distances), length(σs))



distance_ind = 1
sigma_ind = 1
jj = 1


th_pl = Plots.plot()


for d in distances
    global sigma_ind = 1
    for σ in σs
        starting_multipliers, results = optimize_rep_code_evo(d, σ, samples,max_iter)
        # starting_multipliers, results = optimize_rep_code(d, σ, samples,max_iter)
        failure_probs[distance_ind,sigma_ind] = 1 + results.minimum
        failure_probs_no_opt[distance_ind,sigma_ind] = 1 + just_run(d, σ, samples,max_iter)
        global sigma_ind += 1
    end


    plot!(th_pl,σs,failure_probs[distance_ind,:], label="n = $d",markershape=:circle)
    # Access the series objects
    series = th_pl.series_list
    
    # Get the color of the second series
    color1 = series[jj][:seriescolor]

    plot!(th_pl,σs,failure_probs_no_opt[distance_ind,:], label="noOpt, n = $d",markershape=:dtriangle, color=color1)
    plot!(th_pl,matching_results["sigmas"]./sqrt(2π),matching_results["$d"], label="MWPM, n = $d",markershape=:xcross,color = color1)
    display(th_pl)
    Plots.savefig(th_pl,"optimized_th_plot_rep_lsd.pdf")
    
    global distance_ind +=1 
    global jj +=3
end

failure_probs
display(th_pl)
Plots.savefig(th_pl,"optimized_th_plot_rep_lsd.pdf")

using NPZ
npzwrite("results/failure_probabilities_optimized_lsd_CMAES.txt",failure_probs)
npzwrite("results/failure_probabilities_lsd.txt",failure_probs_no_opt)

exit()
#########################################################################
#########################################################################
#########################################################################

# J = symplectic_form(n)
# code = GKP_Rep_Code(n, false, true)
# # logical = vec(code.logical')
# H = code.code[n+1:end,n+1:end]

# G = inv(H)

# tg = initialize_tanner_graph(H)
# #  logical[1] = 1
# #  logical = zeros(Int8, 1, tg.nv)
# pars = (tg, H, G, 0.7, samples, max_iter)

# points = []
# values = []
# for j in 1:5
#     for k in 1:5
#         for l in 1:5
#             push!(points,0.05*[j,k,l]+[1.6,2.,2.])
#             push!(values,cost(0.05*[j,k,l]+[1.6,2.,2.], pars))
#         end
#     end
# end

# points
# values

# using Plots
# Plots.plot(values)

# points[sortperm(values)]