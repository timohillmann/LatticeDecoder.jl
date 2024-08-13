using LatticeDecoder
using Plots
using Graphs
using GraphPlot
using Colors


function gaussian_pdf(x, μ, σ²)
    # Check for valid variance
    if σ² <= 0
        error("Variance must be positive")
    end
    
    # Compute the Gaussian value
    coefficient = 1 / sqrt(2 * π * σ²)
    exponent = -((x - μ)^2) / (2 * σ²)
    
    return coefficient * exp(exponent)
end

function plot_tanner_graph(A::AbstractMatrix)
    m, n = size(A)
    g = SimpleDiGraph(m + n)

    for i in 1:m
        for j in 1:n
            if A[i, j] != 0
                add_edge!(g, i, m + j)
            end
        end
    end
    node_colors = [colorant"lightblue" for _ in 1:(m+n)]
    node_colors[1:m] .= colorant"lightgreen"
    locs_x = vcat(zeros(m), ones(n))
    locs_y = vcat(range(0, 1, length=m), range(0, 1, length=n))
    edge_labels = Dict()
    for e in edges(g)
        i, j = src(e), dst(e)
        if j > m
            edge_labels[e] = string(A[i, j-m])
        end
    end
    
    return gplot(g, locs_x, locs_y, nodefillc=node_colors, nodestrokec=colorant"black", nodestrokelw=1,  edgelabel=edge_labels)
end


# function run_belief_propagation_trace!(tg::LatticeDecoder.TannerGraph, message::Vector{Float64}, σ::Vector{Float64}, max_iter::Int64, decoder::String="lsd"; search_interval::Float64=1.5)

#     if decoder == "nearest"
#         variable_node_iterations! = variable_node_iterations_nearest!
#         decision_step! = decision_step_nearest!

#     elseif decoder == "lsd"
#         variable_node_iterations! = variable_node_iterations_lsd!
#         decision_step! = decision_step_lsd!

#     else
#         error("Invalid decoder. The specified decoder $(decoder) is not supported. Choose either 'nearest' or 'lsd'.")
#     end



#     # set the search interval
#     tg.search_interval = search_interval
#     # println("initialize variable messages")
#     # initilization
#     initialize_messages!(tg, message, σ)

#     messages_trace = Array{Any}(undef, max_iter,length(tg.var_nodes))
#     # basic iteration
#     for i in 1:max_iter
#         # println("starting check node iteration $i")
#         check_node_iterations!(tg)
#         # println("starting th variable node iteration $i")
#         variable_node_iterations!(tg)
#         tg_copy = copy(tg)
#         decision_step!(tg_copy)


#         # add messages to the matrix
#         # for node_idx in 1:length(tg_copy.var_nodes)
#         for node_idx in 1:tg_copy.nv
#             messages_trace[i, node_idx] = tg_copy.var_nodes[node_idx].message
#         end
#     end

#     # final decision
#     decision_step!(tg)
#     for node_idx in 1:tg.nv
#         messages_trace[end, node_idx] = tg.var_nodes[node_idx].message
#     end

#     return messages_trace
# end



function produce_trace(n, σ, max_iter)
        println("n = ", n)
        J = symplectic_form(n)
        code = GKP_Rep_Code(n, false, true)
        # logical = vec(code.logical')
        H = code.code[n+1:end,n+1:end]
        # tg_plot = plot_tanner_graph(H)
        # display(tg_plot)

        G = inv(H)

        tg = initialize_tanner_graph(H)

        y = sample_error(σ, tg.nv)
        return run_belief_propagation_trace!(tg, y, ones(Float64,n) .*(σ), max_iter, "nearest")


end


n = 3
σ = 0.3
iterations = 5


messages_matrix = produce_trace(n, σ, iterations)

size(messages_matrix)

messages_plot = plot(layout=(size(messages_matrix)[1]-1, size(messages_matrix)[2]), legend=false,yaxis = false, size=(200*n, 100*iterations),left_margin=0mm)

for iter in 1:(size(messages_matrix)[1]-1)
    for variable in 1:size(messages_matrix)[2]
        mean = messages_matrix[iter,variable].mean
        var = messages_matrix[iter,variable].var 
        x_values = range(-1, 1, length=2000)
        plot!(messages_plot[iter,variable], x_values, gaussian_pdf.(x_values,mean,var))
        vline!(messages_plot[iter,variable], [messages_matrix[end,variable]]) 
    end
end

display(messages_plot)