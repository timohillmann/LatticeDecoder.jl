# include("tanner_graph.jl")
# include("gaussians.jl")
# include("parallel_bp.jl")

"""
    check_node_message!(vn::VariableNode, tg::TannerGraph, cn_idx::Int64, nb_idx::Int64, vn_pos_idx::Int64)

Compute the new message for a neighbouring variable node of a check node.
"""
function check_node_message!(vn::VariableNode, tg::TannerGraph, cn_idx::Int64, nb_idx::Int64, vn_pos_idx::Int64)
    check_node = tg.check_nodes[cn_idx]

    # compute the average mean and variance of the neighbouring variable nodes
    mean_sum = 0.0
    var_sum = 0.0
    @inbounds @fastmath for i = 1:length(check_node.neighbours)
        if i != vn_pos_idx
            edge_weight = check_node.neighbours[i][2]
            mean_sum += edge_weight * check_node.messages[i].mean
            var_sum += edge_weight^2 * check_node.messages[i].var
        end
    end

    edge_weight = vn.messages[nb_idx].period
    vn.messages[nb_idx].mean = -mean_sum / edge_weight
    vn.messages[nb_idx].var = max(var_sum / edge_weight^2, MIN_VAR)
    vn.messages[nb_idx].period = edge_weight
end


"""
    update_variable_node_lsd!(tg::TannerGraph, vn_idx::Int64)

Update the messages of a variable node in a Tanner graph using the LSD algorithm.
"""
function update_variable_node_lsd!(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    # all check nodes connected to this variable node send their messages to it
    for j = 1:length(vn.neighbours)
        cn_idx, _ = vn.neighbours[j]
        vn_pos_idx = vn.pos_in_check_neighbour[j]
        check_node_message!(vn, tg, cn_idx, j, vn_pos_idx)
    end
    lsd_variable_node_messages!(tg, vn_idx)
end


"""
    update_variable_node_nearest!(tg::TannerGraph, vn_idx::Int64)

Update the messages of a variable node in a Tanner graph using the nearest algorithm.
"""
function update_variable_node_nearest!(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    # all check nodes connected to this variable node send their messages to it
    for j = 1:length(vn.neighbours)
        cn_idx, _ = vn.neighbours[j]
        vn_pos_idx = vn.pos_in_check_neighbour[j]
        check_node_message!(vn, tg, cn_idx, j, vn_pos_idx)
    end
    variable_node_messages_allocationless!(tg, vn_idx)
end





"""
    avg_reliability(vn::VariableNode)

Compute the average reliability of a variable node based on the equation (16) given by 
Ref. Wiriya & Kurkoski, "Reliability-Based Scheduling for Belief Propagation Decoding of LDPC Codes", 2018.
"""
function avg_reliability(vn::VariableNode)
    y = vn.message.mean
    reliability = 0.0
    for msg in vn.messages
        b = -msg.period * (msg.mean - y)
        reliability += 1 / abs(round(b) - b)
    end
    return reliability / length(vn.messages)
end


"""
    avg_reliability!(x::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)

Compute the average reliability of a variable node based on the equation (16) given by
Ref. Wiriya & Kurkoski, "Reliability-Based Scheduling for Belief Propagation Decoding of LDPC Codes", 2018.
"""
function avg_reliability!(x::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)
    @inbounds vn = tg.var_nodes[vn_idx]
    y = vn.message.mean
    reliability = 0.0
    for msg in vn.messages
        b = -msg.period * (msg.mean - y)
        reliability += 1 / abs(round(b) - b)
    end
    @inbounds x[vn_idx] = reliability / length(vn.messages)
end


"""
    update_reliability_schedule!(tg::TannerGraph)

Update the reliability schedule of the variable nodes in a Tanner graph.
"""
function update_reliability_schedule!(tg::TannerGraph)
    @inbounds for i in 1:length(tg.var_nodes)
        avg_reliability!(tg.bp_result, tg, i)
    end
    sortperm!(tg.schedule, tg.bp_result, rev=false)
end

"""
    run_belief_propagation!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Float64, max_iter::Int64)

Run the belief propagation algorithm on a Tanner graph to decode a low-density parity-check (LDPC) code.

# Arguments
- `tg::TannerGraph`: The Tanner graph representing the LDPC code.
- `message::Vector{Float64}`: The message vector obtained from the received codeword.
- `σ::Float64`: The standard deviation of the noise in the received codeword.
- `max_iter::Int64`: The maximum number of iterations to perform.
- `schedule::Vector{Int64}`: Contains the order in which the check and variable nodes are updated.

# Returns
- `bp_result`: The decoded codeword obtained from the belief propagation algorithm.
"""
function run_serial_belief_propagation!(tg::TannerGraph, message::Vector{Float64}, σ::Float64, max_iter::Int64, decoder::String="lsd"; search_interval::Float64=1.5)

    if decoder == "nearest"
        update_variable_node! = update_variable_node_nearest!
        decision_step! = decision_step_nearest!

    elseif decoder == "lsd"
        update_variable_node! = update_variable_node_lsd!
        decision_step! = decision_step_lsd!

    else
        error("Invalid decoder. The specified decoder $(decoder) is not supported. Choose either 'nearest' or 'lsd'.")
    end



    # initilization
    initialize_messages!(tg, message, σ)
    tg.search_interval = search_interval

    # basic iteration
    for i in 1:max_iter
        # printstyled("Iteration: $i\n", color=:blue)
        update_reliability_schedule!(tg)
        for vn_idx in tg.schedule
            update_variable_node!(tg, vn_idx)
        end
    end
    # final decision
    decision_step!(tg)

    return tg.bp_result
end

# run_serial_belief_propagation!(tg::TannerGraph, message::Vector{Float64}, σ::Float64, max_iter::Int64, ) = run_serial_belief_propagation!(tg, message, σ, max_iter)



##################################################################
##################################################################
##################################################################


function run_serial_belief_propagation_trace!(tg::LatticeDecoder.TannerGraph, message::Vector{Float64}, σ::Vector{Float64}, max_iter::Int64, decoder::String="lsd"; search_interval::Float64=1.5)

    if decoder == "nearest"
        update_variable_node! = update_variable_node_nearest!
        decision_step! = decision_step_nearest!
        simulate_decision = variable_node_decision_simulated_nearest

    elseif decoder == "lsd"
        update_variable_node! = update_variable_node_lsd!
        decision_step! = decision_step_lsd!
        simulate_decision = variable_node_decision_simulated_lsd

    else
        error("Invalid decoder. The specified decoder $(decoder) is not supported. Choose either 'nearest' or 'lsd'.")
    end



    # initilization
    initialize_messages!(tg, message, σ)
    tg.search_interval = search_interval


    messages_trace = Array{Any}(undef, max_iter + 2, length(tg.var_nodes))

    for node_idx in 1:tg.nv
        messages_trace[1, node_idx] = simulate_decision(tg, node_idx)
    end

    # basic iteration
    for i in 1:max_iter
        # printstyled("Iteration: $i\n", color=:blue)
        update_reliability_schedule!(tg)
        for vn_idx in tg.schedule
            update_variable_node!(tg, vn_idx)
        end

        for node_idx in 1:tg.nv
            messages_trace[i+1, node_idx] = simulate_decision(tg, node_idx)
        end

    end
    # final decision
    for node_idx in 1:tg.nv
        messages_trace[end-1, node_idx] = simulate_decision(tg, node_idx)
    end

    decision_step!(tg)
    messages_trace[end, :] = message



    # println("bp result: ", tg.bp_result)

    return messages_trace, tg.bp_result
end