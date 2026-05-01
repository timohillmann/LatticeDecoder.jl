using LatticeDecoder
using Plots
using Colors
using LaTeXStrings
using Measures
using NPZ

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
        code = GKP_Rep_Code(n, false, true)
        # logical = vec(code.logical')
        H = code.code[n+1:end,n+1:end]
        H[1,1] = 1/sqrt(2)
        # H[1,end]=1/sqrt(2)
        H[1,3]=1/sqrt(2)
        display(sqrt(2)*H)
        # tg_plot = plot_tanner_graph(H)
        # display(tg_plot)

        G = inv(H)

        tg = initialize_tanner_graph(H)

        # y = sample_error(σ, tg.nv) 
        # y += rand((0, 1), tg.nv) ./ sqrt(2)
        y = [0.555659614502603, 0.2797015135834352, 1.0083195669887157 - sqrt(2)]
        # println("y = $y")
        
        messages_matrix,bp_result = LatticeDecoder.run_belief_propagation_trace!(tg, y, ones(Float64,n) .*(σ), max_iter, "lsd")
        decoded = G*hard_decision(bp_result,H)
        return messages_matrix, bp_result, decoded
        

end


n = 3
σ = 0.7 /(2*sqrt(pi))
iterations = 5


#  run_serial_belief_propagation_trace!
# run_belief_propagation_trace!

messages_matrix,bp_result, decoded = produce_trace(n, σ, iterations);

messages = messages_matrix[1:end-1, :]
means = zeros((size(messages)))
vars = zeros(size(messages))

for index in eachindex(messages)
    means[index] = messages[index].mean
    vars[index] = messages[index].var
end

npzwrite("/Users/timo/Documents/LatticeDecoder.jl/data/timeseries/trace_timeseries/rep_code_3_lsd_balanced_last_v3.npz", Dict("means" => means, "vars" => vars, "bp_result" => bp_result, "decoded" => decoded))


function plot_bp_messages(messages_matrix)
    iterations = size(messages_matrix)[1]-2
    nvars = size(messages_matrix)[2]
    
    tick_vals = [-sqrt(2), -sqrt(2)/2, 0, sqrt(2)/2, sqrt(2)]
    tick_labels = [L"-\sqrt{2}",L"-\sqrt{2}/2",L"0",L"\sqrt{2}/2",L"\sqrt{2}"]
    ticks = (tick_vals, tick_labels )
    messages_plot = plot(layout=grid(iterations+1, nvars),legend=false, yaxis = false, size=(300*nvars, 200*iterations), xticks=ticks, left_margin=10mm, bottom_margin=10mm)
    for iter in 1:(iterations+1)
        for variable in 1:nvars
            mean = messages_matrix[iter,variable].mean
            var = messages_matrix[iter,variable].var 
            x_values = range(-sqrt(2), sqrt(2), length=2000)
            plot!(messages_plot[iter,variable], x_values, gaussian_pdf.(x_values,mean,var),linewidth=1.7)
            if variable == 1
                ylabel!(messages_plot[iter,variable],"it $(iter-1)")
            end
            if iter == 1
                title!(messages_plot[iter,variable],"var $(variable)")
            end
            vline!(messages_plot[iter,variable], [messages_matrix[end,variable]],linewidth=1.7) 
        end
    end

    for variable in 1:size(messages_matrix)[2]
        vline!(messages_plot[iterations+1,variable], [Float64(decoded[variable])],linewidth=2.2, linestyle=:dash) 
    end

    return messages_plot
end


display(plot_bp_messages(messages_matrix))

#TODO: I would like to check whether BP solves CVP, at least most of the time, 


