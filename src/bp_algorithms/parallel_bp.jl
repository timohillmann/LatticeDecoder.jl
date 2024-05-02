include("tanner_graph.jl")
include("list_sphere_decoder_gaussians.jl")

VarNodeAlloc = FourGaussianAlloc(gaussian(0.0, 0.5), gaussian(0.0, 0.5), gaussian(0.0, 0.5), gaussian(0.0, 0.5))

"""
    initialize_messages!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Float64)

Initialize the messages of the Tanner graph for LDLC decoding.
"""
function initialize_messages!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Float64)

    # Set variable node messages to received channel message
    for i in 1:length(tg.var_nodes)
        tg.var_nodes[i].message.mean = syndrome[i]
        tg.var_nodes[i].message.var = σ^2
    end

    # Collect the messages from the neighbouring variable nodes
    for i in 1:length(tg.check_nodes)
        check_node = tg.check_nodes[i]

        for idx in 1:length(check_node.neighbours)
            vn_idx, edge_weight = check_node.neighbours[idx]
            check_node.messages[idx].mean = 1.0 * tg.var_nodes[vn_idx].message.mean
            check_node.messages[idx].var = 1.0 * tg.var_nodes[vn_idx].message.var
        end
    end
end

"""
    check_node_iterations!(tg::TannerGraph)

    Iterates over all check nodes and updates the messages of the variable nodes.
"""
function check_node_iterations!(tg::TannerGraph)
    for i in 1:length(tg.check_nodes)
        check_node_messages!(tg, i)
    end
end


"""
    check_node_messages!(tg::TannerGraph, cn_idx::Int64)

    Compute the new messages for the neighbouring variable nodes of a check node with index `cn_idx`.
"""
function check_node_messages!(tg::TannerGraph, cn_idx::Int64)
    check_node = tg.check_nodes[cn_idx]

    # compute the average mean and variance of the neighbouring variable nodes
    mean_sum = 0.0
    var_sum = 0.0
    @inbounds @fastmath for i = 1:length(check_node.neighbours)
        edge_weight = check_node.neighbours[i][2]
        mean_sum += edge_weight * check_node.messages[i].mean
        var_sum += edge_weight^2 * check_node.messages[i].var
    end

    # compute the new messages for the neighbouring variable nodes
    @inbounds for i = 1:length(check_node.neighbours)
        vn_idx, edge_weight = check_node.neighbours[i]
        idx = check_node.pos_in_var_neighbour[i]
        vn = tg.var_nodes[vn_idx]
        vn.messages[idx].mean = -(mean_sum - check_node.messages[i].mean * edge_weight) / edge_weight
        vn.messages[idx].var = max((var_sum - check_node.messages[i].var * edge_weight^2) / edge_weight^2, MIN_VAR)
        vn.messages[idx].period = edge_weight
    end
end


"""
    variable_node_iterations!(tg::TannerGraph)

    Iterates over all variable nodes and updates the messages of the check nodes.
"""
function variable_node_iterations!(tg::TannerGraph)
    for i in 1:length(tg.var_nodes)
        # variable_node_messages!(tg, i) # Liu paper style
        # variable_node_messages_allocationless!(tg, i)
        # mm_variable_node_messages!(tg, i)
        lsd_variable_node_messages!(tg, i)
    end
end


"""
    variable_node_messages!(tg::TannerGraph, vn_idx::Int64)

    Compute the new messages for the neighbouring check nodes of a variable node with index `vn_idx`.
    This implementation uses the sum of two Gaussian distributions to compute the new messages.
    It is due to Liu et al., Efficient Decoding of Low Density Lattice Codes, IEEE WCL 8 4 (2019).
"""
function variable_node_messages!(tg::TannerGraph, vn_idx::Int64)
    var_node = tg.var_nodes[vn_idx]

    # multiply together all the messages from the neighbouring check nodes
    # N(w, y_k, simga_k^2) * prod((N_{L, j} + N_{R, j}), j != k)
    for j = 1:length(var_node.neighbours)
        cn_idx, edge_weight = var_node.neighbours[j]
        idx = var_node.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]

        gL = gaussian(var_node.message.mean, var_node.message.var)
        gR = gaussian(var_node.message.mean, var_node.message.var)

        for i = 1:length(var_node.messages)
            if i != j  # don't include the message from the current check node
                g1, g2 = nearest(var_node.messages[i], var_node.message.mean, var_node.messages[i].period, 1.5)
                prod!(gL, g1)
                prod!(gR, g2)
            end
        end
        cn.messages[idx] = sum(gL, gR)
    end
end

function variable_node_messages_allocationless!(tg::TannerGraph, vn_idx::Int64, Alloc::FourGaussianAlloc)
    var_node = tg.var_nodes[vn_idx]

    # multiply together all the messages from the neighbouring check nodes
    # N(w, y_k, simga_k^2) * prod((N_{L, j} + N_{R, j}), j != k)
    for j = 1:length(var_node.neighbours)
        cn_idx, edge_weight = var_node.neighbours[j]
        idx = var_node.pos_in_check_neighbour[j]

        Alloc.gL.mean = var_node.message.mean
        Alloc.gR.mean = var_node.message.mean
        Alloc.gL.var = var_node.message.var
        Alloc.gR.var = var_node.message.var
        Alloc.gL.weight = 1.0
        Alloc.gR.weight = 1.0

        for i = 1:length(var_node.messages)
            if i != j  # don't include the message from the current check node
                nearest!(Alloc.g1, Alloc.g2, var_node.messages[i], var_node.message.mean, 1.5)
                prod!(Alloc.gL, Alloc.g1)
                prod!(Alloc.gR, Alloc.g2)

            end
        end
        sum!(tg.check_nodes[cn_idx].messages[idx], Alloc.gL, Alloc.gR)
    end
end

variable_node_messages_allocationless!(tg::TannerGraph, vn_idx::Int64) = variable_node_messages_allocationless!(tg, vn_idx, VarNodeAlloc)

"""
    decision_step(tg::TannerGraph)

    Compute the decision for each variable node in the Tanner graph.

"""
function decision_step(tg::TannerGraph)
    for i in 1:length(tg.var_nodes)
        # variable_node_decision!(tg.bp_result, tg, i)
        variable_node_decision_allocationless!(tg.bp_result, tg, i)
    end
end


"""
    variable_node_decision!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)

    Compute the decision for a variable node with index `vn_idx` in the Tanner graph.
    This implementation uses the sum of two Gaussian distributions to compute the decision.
    It is due to Liu et al., Efficient Decoding of Low Density Lattice Codes, IEEE WCL 8 4 (2019).
"""
function variable_node_decision!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]

    gL = gaussian(vn.message.mean, vn.message.var)
    gR = gaussian(vn.message.mean, vn.message.var)

    for i = 1:length(vn.messages)
        cn_idx, edge_weight = vn.neighbours[i]
        g1, g2 = nearest(vn.messages[i], vn.message.mean, edge_weight, 1.5)
        prod!(gL, g1)
        prod!(gR, g2)
    end

    vn.message = sum(gL, gR)

    bp_result[vn_idx] = vn.message.mean
end


function variable_node_decision_allocationless!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64, Alloc::FourGaussianAlloc)
    vn = tg.var_nodes[vn_idx]

    Alloc.gL.mean = vn.message.mean
    Alloc.gR.mean = vn.message.mean
    Alloc.gL.var = vn.message.var
    Alloc.gR.var = vn.message.var
    Alloc.gL.weight = 1.0
    Alloc.gR.weight = 1.0


    for i = 1:length(vn.messages)
        nearest!(Alloc.g1, Alloc.g2, vn.messages[i], vn.message.mean, 1.5)
        prod!(Alloc.gL, Alloc.g1)
        prod!(Alloc.gR, Alloc.g2)
    end

    sum!(vn.message, Alloc.gL, Alloc.gR)
    bp_result[vn_idx] = vn.message.mean
end

function variable_node_mother_message!(tg::TannerGraph, vn_idx::Int64, Alloc::FourGaussianAlloc)
    vn = tg.var_nodes[vn_idx]

    Alloc.gL.mean = vn.message.mean
    Alloc.gR.mean = vn.message.mean
    Alloc.gL.var = vn.message.var
    Alloc.gR.var = vn.message.var
    Alloc.gL.weight = 1.0
    Alloc.gR.weight = 1.0

    for i = 1:length(vn.messages)
        nearest!(Alloc.g1, Alloc.g2, vn.messages[i], vn.message.mean, 1.5)
        prod!(Alloc.gL, Alloc.g1)
        prod!(Alloc.gR, Alloc.g2)
    end
end


function mm_variable_node_messages!(tg::TannerGraph, vn_idx::Int64, Alloc::FourGaussianAlloc)
    vn = tg.var_nodes[vn_idx]

    variable_node_mother_message!(tg, vn_idx, Alloc)

    for j = 1:length(vn.neighbours)
        cn_idx, edge_weight = vn.neighbours[j]
        idx = vn.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]
        nearest!(Alloc.g1, Alloc.g2, vn.messages[j], vn.message.mean, vn.messages[j].period)
        gL_j = divide(Alloc.gL, Alloc.g1)
        gR_j = divide(Alloc.gR, Alloc.g2)
        sum!(tg.check_nodes[cn_idx].messages[idx], gL_j, gR_j)
    end

end

mm_variable_node_messages!(tg::TannerGraph, vn_idx::Int64) = mm_variable_node_messages!(tg, vn_idx, VarNodeAlloc)

variable_node_decision_allocationless!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64) = variable_node_decision_allocationless!(bp_result, tg, vn_idx, VarNodeAlloc)


"""
    run_belief_propagation!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Float64, max_iter::Int64)

Run the belief propagation algorithm on a Tanner graph to decode a low-density parity-check (LDPC) code.

# Arguments
- `tg::TannerGraph`: The Tanner graph representing the LDPC code.
- `message::Vector{Float64}`: The message vector obtained from the received codeword.
- `σ::Float64`: The standard deviation of the noise in the received codeword.
- `max_iter::Int64`: The maximum number of iterations to perform.

# Returns
- `bp_result`: The decoded codeword obtained from the belief propagation algorithm.
"""
function run_belief_propagation!(tg::TannerGraph, message::Vector{Float64}, σ::Float64, max_iter::Int64)
    # initilization
    initialize_messages!(tg, message, σ)

    # basic iteration
    for i in 1:max_iter
        check_node_iterations!(tg)
        variable_node_iterations!(tg)
    end

    # final decision
    decision_step(tg)

    return tg.bp_result
end


# function hard_decision(bp_result::Vector{Float64}, H::AbstractArray)
#     return Int64.(round.(H * bp_result))
# end``