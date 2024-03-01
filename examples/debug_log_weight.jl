# using LatticeDecoder
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/parallel_bp_log_weight.jl");
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/tanner_graph_log_weight.jl");
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/code_constructors/classical_ldlc.jl");
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/gaussians_log_weight.jl");
d = 5;
n = 961;

H = classical_ldlc(d, n, true);

tg = initialize_tanner_graph(H);


function extract_mean!(out::AbstractArray{Float64}, tg, iteration)
    for i in 1:length(tg.var_nodes)
        vn = tg.var_nodes[i]
        pos = vn.pos_in_check_neighbour[1]
        cn, weight = vn.neighbours[1]
        out[i, iteration] = tg.check_nodes[cn].messages[pos].mean
    end
end

function extract_variance!(out::AbstractArray{Float64}, tg, iteration::Int64, σ::Float64)
    for i in 1:length(tg.var_nodes)
        vn = tg.var_nodes[i]
        pos = vn.pos_in_check_neighbour[1]
        cn, weight = vn.neighbours[1]
        out[i, iteration] = tg.check_nodes[cn].messages[pos].var / σ^2

    end

end


σ = 0.241
message = σ * randn(length(tg.var_nodes));
max_iter = 30
var = zeros(length(tg.var_nodes), max_iter);
means = zeros(length(tg.var_nodes), max_iter);

# initilization
initialize_messages!(tg, message, σ)

# # basic iteration
for i in 1:max_iter
    check_node_iterations!(tg)
    variable_node_iterations!(tg)
    extract_variance!(var, tg, i, σ)
    extract_mean!(means, tg, i)
end
using Plots

# plot variance of message vs iteration
fig = plot(var', label="", xlabel="iteration", ylabel="variance of message", title="Variance of message vs iteration")
# cut of plot y axis at 0.5
fig = plot!(fig, ylims=(5e-5, 0.75))

# plot 1/(1.6 * iteration)
plot!(1 ./ (1.6 * (1:max_iter)), label="1/(1.6 * iteration)", lw=2,
    marker=:circle, markersize=5, yscale=:log10)

#plot horizontal line at 2/3
plot!(2 / 3 * ones(max_iter), label="2/3", lw=2, linestyle=:dash, color=:black)


# make new figure for the means
fig = plot()
# plot means of message vs iteration
fig = plot(means', label="", xlabel="iteration", ylabel="mean of message", title="Mean of message vs iteration")

using BenchmarkTools

function basic_iteration(tg, max_iter)
    for i in 1:max_iter
        check_node_iterations!(tg)
        variable_node_iterations!(tg)
    end
end

@benchmark basic_iteration(tg, 15)

@benchmark check_node_iterations!(tg)

@benchmark check_node_messages!(tg, 1)

# @benchmark variable_node_iterations!(tg)
@benchmark variable_node_messages_allocationless!(tg, 1)

@benchmark decision_step(tg)

@benchmark variable_node_decision_allocationless!(tg.bp_result, tg, 1)

x = 1

# set yscale log
# fig = plot!(fig, yscale=:log10)

# show the figure
# fig

# using LinearAlgebra
# include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/parallel_bp.jl");
# include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/tanner_graph.jl");
# include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/code_constructors/classical_ldlc.jl");


# H = [0 -0.8 0 -0.5 1.0 0;
#     0.8 0.0 0.0 1.0 0.0 -0.5;
#     0.0 0.5 1.0 0.0 0.8 0;
#     0.0 0.0 -0.5 -0.8 0.0 1.0;
#     1.0 0.0 0.0 0.0 0.5 0.8;
#     0.5 -1.0 -0.8 0.0 0.0 0.0];
# using LinearAlgebra;
# H = H / abs(det((H)))^(-1 / 6);
# pe = [-0.2, 0.01, -0.1, 0.16, 0.84, 0.18];

# n = 6;
# d = 3;
# σ = 0.23;
# iterations = 25;
# tg = initialize_tanner_graph(H);
# initialize_messages!(tg, pe, σ);

# println("Initialized messages")
# for vn in tg.var_nodes
#     println(vn.message)
# end
# println("Messages from Variable Node to Check Node")
# for cn in tg.check_nodes
#     println(cn.messages)
# end

# println("Perform Check Node Iteration 1")
# check_node_iterations!(tg)

# println("Messages from Check Nodes to Variable Node")
# for vn in tg.var_nodes
#     # for msg in vn.messages
#     #     println(msg)
#     # end
#     println(vn.messages)
# end

# println("Perform Variable Node Iteration 1")
# variable_node_iterations!(tg)

# println("Messages from Variable Node to Check Node")
# for cn in tg.check_nodes
#     println(cn.messages)
# end
# println("Perform Check Node Iteration 2")
# check_node_iterations!(tg)

# println("Messages from Check Nodes to Variable Node")
# for vn in tg.var_nodes
#     println(vn.messages)
# end

# println("Perform Variable Node Iteration 2")
# variable_node_iterations!(tg)

# println("Messages from Variable Node to Check Node")
# for cn in tg.check_nodes
#     println(cn.messages)
# end

# println("Decision Step")
# decision_step(tg)

# println("Soft Decision")
# println(tg.bp_result)
