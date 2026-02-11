include("tanner_graph_log_weight.jl")
include("list_sphere_decoder_log_weight.jl")

VarNodeAlloc = FourGaussianLogAlloc(gaussian_log_weight(0.0, 0.5))

MotherNodeAlloc = SixGaussianLogAlloc(gaussian_log_weight(0.0, 0.5))

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
    initialize_messages!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Vector{Float64})

Initialize the messages of the Tanner graph for LDLC decoding. NOTE: here the variances can be different for different variable nodes
"""
function initialize_messages!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Vector{Float64})

    # Set variable node messages to received channel message
    for i in 1:length(tg.var_nodes)
        tg.var_nodes[i].message.mean = syndrome[i]
        tg.var_nodes[i].message.var = σ[i]^2
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
    for i in 1:tg.nc
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
    @inbounds for i = 1:length(check_node.neighbours)
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
    variable_node_iterations_nearest!(tg::TannerGraph)

    Iterates over all variable nodes and updates the messages of the check nodes using the nearest algorithm.
"""

function variable_node_iterations_nearest!(tg::TannerGraph)
    for i in 1:tg.nv
        variable_node_messages_allocationless!(tg, i)
    end
end


"""
    variable_node_iterations_lsd!(tg::TannerGraph)

    Iterates over all variable nodes and updates the messages of the check nodes using the LSD algorithm.
"""
function variable_node_iterations_lsd!(tg::TannerGraph)
    for i in 1:tg.nv
        lsd_variable_node_messages!(tg, i)
    end
end


function variable_node_iterations_M_gaussian!(tg::TannerGraph, M::Int64)
    for i = 1:tg.nv
        variable_node_messages_M_gaussian!(tg, i, M)
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

        gL = gaussian_log_weight(var_node.message.mean, var_node.message.var)
        gR = gaussian_log_weight(var_node.message.mean, var_node.message.var)

        for i = 1:length(var_node.messages)
            if i != j  # don't include the message from the current check node
                g1, g2 = nearest(var_node.messages[i], var_node.message.mean, var_node.messages[i].period, tg.search_interval)
                prod!(gL, g1)
                prod!(gR, g2)
            end
        end
        cn.messages[idx] = sum(gL, gR)
    end
end




"""
    forward_backward_decoder(mixtures, g)

Compute variable node outputs using forward-backward recursion for M-Gaussian LDLC decoding.

Arguments:
- mixtures::Vector{Vector{gaussian_log_weight}}: Each element is a mixture from a check node (M Gaussians).
- g::gaussian_log_weight: Channel Gaussian for the variable node.

Returns:
- outputs::Vector{gaussian_log_weight}: Single Gaussian per edge after moment matching.
"""
function forward_backward_recursion(mixtures::Vector{Vector{gaussian_log_weight}}, g::gaussian_log_weight)
    d = length(mixtures)  # degree of variable node

    # Forward recursion
    forward = Vector{Vector{gaussian_log_weight}}(undef, d)
    forward[1] = mixtures[1]
    for i in 2:d
        forward[i] = prod(forward[i-1], mixtures[i])
    end

    # Backward recursion
    backward = Vector{Vector{gaussian_log_weight}}(undef, d)
    backward[d] = mixtures[d]
    for i in (d-1):-1:1
        backward[i] = prod(backward[i+1], mixtures[i])
    end

    # Compute outputs for each edge
    outputs = Vector{gaussian_log_weight}([gaussian_log_weight(0.0, 1.0) for _ = 1:d])
    for i in 1:d
        # Combine forward and backward excluding mixtures[i]
        # left = (i > 1) ? forward[i-1] : nothing
        # right = (i < d) ? backward[i+1] : nothing
        if i == 1
            combined = prod(backward[i+1], g)
        elseif i == d
            combined = prod(forward[i-1], g)
        else
            combined = prod(forward[i-1], backward[i+1])
            prod!(combined, g)  # multiply channel last
        end
        moment_matching!(outputs[i], combined)
    end

    return outputs
end



function variable_node_messages_M_gaussian!(tg::TannerGraph, vn_idx::Int64, M::Int64)
    var_node = tg.var_nodes[vn_idx]
    d = length(var_node.neighbours)
    if d == 1
        # special case for degree 1 variable nodes
        cn_idx, edge_weight = var_node.neighbours[1]
        idx = var_node.pos_in_check_neighbour[1]
        cn = tg.check_nodes[cn_idx]
        cn.messages[idx] = copy(var_node.message)
        return
    else
        mixtures = [Vector{gaussian_log_weight}(undef, M) for _ in 1:d]
        @inbounds for j = 1:d
            cn_idx, edge_weight = var_node.neighbours[j]    
            mixtures[j] = m_nearest(var_node.messages[j], var_node.message.mean, edge_weight, M)
        end
            outputs = forward_backward_recursion(mixtures, var_node.message)
    
        for j = 1:d
            cn_idx, _ = var_node.neighbours[j]
            idx = var_node.pos_in_check_neighbour[j]
            cn = tg.check_nodes[cn_idx]
    
            cn.messages[idx].mean = outputs[j].mean
            cn.messages[idx].var = outputs[j].var
        end
    end

end

function variable_node_decision_M_gaussian!(tg::TannerGraph, vn_idx::Int64, M::Int64)
    var_node = tg.var_nodes[vn_idx]
    d = length(var_node.neighbours)
    mixtures = [Vector{gaussian_log_weight}(undef, M) for _ in 1:d]
    @inbounds for j = 1:d
        cn_idx, edge_weight = var_node.neighbours[j]    
        mixtures[j] = m_nearest(var_node.messages[j], var_node.message.mean, edge_weight, M)
    end
    
    out = deepcopy(mixtures[1])
    for j = 2:d
        out = prod(out, mixtures[j])
    end
    out = prod(out, var_node.message)
    moment_matching!(var_node.message, out)
    tg.bp_result[vn_idx] = var_node.message.mean
end    


function decision_step_M_gaussian!(tg::TannerGraph, M::Int)
    for vn = 1:tg.nv
        variable_node_decision_M_gaussian!(tg, vn, M)
    end
end


function variable_node_messages_allocationless!(tg::TannerGraph, vn_idx::Int64, Alloc::FourGaussianLogAlloc)
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
        Alloc.gL.log_weight = 0.0
        Alloc.gR.log_weight = 0.0

        for i = 1:length(var_node.messages)
            if i != j  # don't include the message from the current check node
                nearest!(Alloc.g1, Alloc.g2, var_node.messages[i], var_node.message.mean, tg.search_interval)
                prod!(Alloc.gL, Alloc.g1)
                prod!(Alloc.gR, Alloc.g2)

            end
        end
        sum!(tg.check_nodes[cn_idx].messages[idx], Alloc.gL, Alloc.gR)
    end
end

variable_node_messages_allocationless!(tg::TannerGraph, vn_idx::Int64) = variable_node_messages_allocationless!(tg, vn_idx, VarNodeAlloc)

"""
    decision_step_nearest!(tg::TannerGraph)

    Compute the decision for each variable node in the Tanner graph using the nearest algorithm.

"""
function decision_step_nearest!(tg::TannerGraph)
    for i in 1:tg.nv
        variable_node_decision_allocationless!(tg.bp_result, tg, i)
    end
end


"""
    decision_step_lsd!(tg::TannerGraph)

    Compute the decision for each variable node in the Tanner graph using the LSD algorithm.

"""
function decision_step_lsd!(tg::TannerGraph)
    for i in 1:tg.nv
        _lsd_variable_node_decision!(tg, i)
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

    gL = gaussian_log_weight(vn.message.mean, vn.message.var)
    gR = gaussian_log_weight(vn.message.mean, vn.message.var)

    for i = 1:length(vn.messages)
        cn_idx, edge_weight = vn.neighbours[i]
        g1, g2 = nearest(vn.messages[i], vn.message.mean, edge_weight, tg.search_interval)
        prod!(gL, g1)
        prod!(gR, g2)
    end

    vn.message = sum(gL, gR)

    bp_result[vn_idx] = vn.message.mean
end


function variable_node_decision_allocationless!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64, Alloc::FourGaussianLogAlloc)
    vn = tg.var_nodes[vn_idx]

    Alloc.gL.mean = vn.message.mean
    Alloc.gR.mean = vn.message.mean
    Alloc.gL.var = vn.message.var
    Alloc.gR.var = vn.message.var
    Alloc.gL.log_weight = 0.0
    Alloc.gR.log_weight = 0.0


    for i = 1:length(vn.messages)
        nearest!(Alloc.g1, Alloc.g2, vn.messages[i], vn.message.mean, tg.search_interval)
        prod!(Alloc.gL, Alloc.g1)
        prod!(Alloc.gR, Alloc.g2)
    end

    sum!(vn.message, Alloc.gL, Alloc.gR)
    bp_result[vn_idx] = vn.message.mean
end

variable_node_decision_allocationless!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64) = variable_node_decision_allocationless!(bp_result, tg, vn_idx, VarNodeAlloc)


function variable_node_mother_message!(tg::TannerGraph, vn_idx::Int64, Alloc::SixGaussianLogAlloc)
    vn = tg.var_nodes[vn_idx]

    Alloc.gL.mean = vn.message.mean
    Alloc.gR.mean = vn.message.mean
    Alloc.gL.var = vn.message.var
    Alloc.gR.var = vn.message.var
    Alloc.gL.log_weight = 0.0
    Alloc.gR.log_weight = 0.0



    for i = 1:length(vn.messages)
        nearest!(Alloc.g1, Alloc.g2, vn.messages[i], vn.message.mean, tg.search_interval)
        prod!(Alloc.gL, Alloc.g1)
        prod!(Alloc.gR, Alloc.g2)
    end
end


function mm_variable_node_messages!(tg::TannerGraph, vn_idx::Int64, Alloc::SixGaussianLogAlloc)
    vn = tg.var_nodes[vn_idx]

    variable_node_mother_message!(tg, vn_idx, Alloc)

    for j = 1:length(vn.neighbours)
        cn_idx, edge_weight = vn.neighbours[j]
        idx = vn.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]
        nearest!(Alloc.g1, Alloc.g2, vn.messages[j], vn.message.mean, tg.search_interval)
        divide!(Alloc.gL_j, Alloc.gL, Alloc.g1)
        divide!(Alloc.gR_j, Alloc.gR, Alloc.g2)
        sum!(tg.check_nodes[cn_idx].messages[idx], Alloc.gL_j, Alloc.gR_j)
    end

end

mm_variable_node_messages!(tg::TannerGraph, vn_idx::Int64) = mm_variable_node_messages!(tg, vn_idx, MotherNodeAlloc)


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
function run_belief_propagation!(tg::TannerGraph, message::Vector{Float64}, σ::Float64, max_iter::Int64, decoder::Union{String, Int64}="lsd"; search_interval::Float64=1.5)

    if decoder == "nearest"
        variable_node_iterations! = variable_node_iterations_nearest!
        decision_step! = decision_step_nearest!

    elseif decoder == "lsd"
        variable_node_iterations! = variable_node_iterations_lsd!
        decision_step! = decision_step_lsd!


    elseif typeof(decoder) == Int64
        # Pass decoder as an extra argument
        # variable_node_iterations_M_gaussian!(tg::TannerGraph) = variable_node_iterations_M_gaussian!(tg, decoder)
        variable_node_iterations! = (args...) -> variable_node_iterations_M_gaussian!(args..., decoder)
        decision_step! = (args...) -> variable_node_decision_M_gaussian!(args..., decoder)

    else
        error("Invalid decoder. The specified decoder $(decoder) is not supported. Choose either 'nearest' or 'lsd'.")
    end



    # set the search interval
    tg.search_interval = search_interval
    # println("initialize variable messages")
    # initilization
    initialize_messages!(tg, message, σ)


    # basic iteration
    for i in 1:max_iter
        # println("starting check node iteration $i")
        check_node_iterations!(tg)
        # println("starting th variable node iteration $i")
        variable_node_iterations!(tg)
    end

    # final decision
    decision_step!(tg)

    return tg.bp_result
end


##################################################################
##################################################################
##################################################################

function variable_node_decision_simulated_nearest(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]

    gL = gaussian_log_weight(vn.message.mean, vn.message.var)
    gR = gaussian_log_weight(vn.message.mean, vn.message.var)

    for i = 1:length(vn.messages)
        cn_idx, edge_weight = vn.neighbours[i]
        g1, g2 = nearest(vn.messages[i], vn.message.mean, edge_weight, tg.search_interval)
        prod!(gL, g1)
        prod!(gR, g2)
    end

    return sum(gL, gR)
end

function variable_node_decision_simulated_lsd(tg::TannerGraph, vn_idx::Int64)
    # println("LSD Variable Node Decision")
    vn = tg.var_nodes[vn_idx]
    msg_vector = _collect_msg_vector(vn)
    lsd_inputs = ListSphereDecodingInput(msg_vector)
    L, D = simplified_lsd(lsd_inputs)
    candidate_gaussians = _calculate_candidate_gaussians(lsd_inputs, L, D, msg_vector)
    # message = copy(vn.message)
    # moment_matching!(message, candidate_gaussians)
    # return message
    return moment_matching(candidate_gaussians)
end



function run_belief_propagation_trace!(tg::LatticeDecoder.TannerGraph, message::Vector{Float64}, σ::Vector{Float64}, max_iter::Int64, decoder::String="lsd"; search_interval::Float64=1.5)

    if decoder == "nearest"
        variable_node_iterations! = variable_node_iterations_nearest!
        decision_step! = decision_step_nearest!
        simulate_decision = variable_node_decision_simulated_nearest

    elseif decoder == "lsd"
        variable_node_iterations! = variable_node_iterations_lsd!
        decision_step! = decision_step_lsd!
        simulate_decision = variable_node_decision_simulated_lsd

    else
        error("Invalid decoder. The specified decoder $(decoder) is not supported. Choose either 'nearest' or 'lsd'.")
    end



    # set the search interval
    tg.search_interval = search_interval
    # println("initialize variable messages")
    # initilization
    initialize_messages!(tg, message, σ)

    messages_trace = Array{Any}(undef, max_iter + 2, length(tg.var_nodes))

    for node_idx in 1:tg.nv
        messages_trace[1, node_idx] = simulate_decision(tg, node_idx)
    end

    # basic iteration
    for i in 1:max_iter
        # println("starting check node iteration $i")
        check_node_iterations!(tg)
        # println("starting th variable node iteration $i")
        variable_node_iterations!(tg)

        # add messages to the matrix
        # for node_idx in 1:length(tg_copy.var_nodes)
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



####################
####################



# -------------------------------
# Allocation Structures
# -------------------------------



mutable struct FBAlloc
    forward::Vector{Vector{gaussian_log_weight}} 
    backward::Vector{Vector{gaussian_log_weight}}
    mixtures::Vector{Vector{gaussian_log_weight}}
    combined::Vector{gaussian_log_weight}
    outputs::Vector{gaussian_log_weight}
    ws_alloc::Vector{Float64}
end

function FBAlloc(d::Int, M::Int)
    mixtures = [map(copy, fill(gaussian_log_weight(0.0, 1.0), M)) for _ in 1:d]
    forward = [map(copy, fill(gaussian_log_weight(0.0, 1.0), M^k)) for k = 1:d]
    backward = [map(copy, fill(gaussian_log_weight(0.0, 1.0), M^(k))) for k = d:-1:1]
    combined = map(copy, fill(gaussian_log_weight(0.0, 1.0), M^(d-1)))
    outputs = map(copy, fill(gaussian_log_weight(0.0, 1.0), d))
    ws_alloc = Vector{Float64}(undef, M^(d-1))
    return FBAlloc(forward, backward, mixtures, combined, outputs, ws_alloc)
end

mutable struct LDLCDecoder
    tg::TannerGraph
    allocs::Dict{Int, FBAlloc}  # keyed by degree
    M::Int64
end


"""
    LDLCDecoder(tg, M)

Initialize LDLC decoder with preallocated buffers for forward-backward recursion.
"""
function LDLCDecoder(tg::TannerGraph, M::Int)
    degrees = unique(length(vn.neighbours) for vn in tg.var_nodes)
    allocs = Dict(d => FBAlloc(d, M) for d in degrees)
    return LDLCDecoder(tg, allocs, M)
end



"""
    variable_node_iterations!(dec::LDLCDecoder)

Runs the variable node update step for all variable nodes in the decoder `dec` for one iteration.
For each variable node in the Tanner graph, performs the allocation-less M-gaussian message update
and writes outgoing messages for the current BP iteration.
"""
function variable_node_iterations!(dec::LDLCDecoder)
    for i = 1:dec.tg.nv
        variable_node_messages_M_gaussian_allocationless!(dec.tg, i, dec)
    end
end

"""
    variable_node_messages_M_gaussian_allocationless!(tg::TannerGraph, vn_idx::Int64, decoder::LDLCDecoder)

Perform the variable node update for a single variable node, identified by `vn_idx`,
in the Tanner graph `tg` using the specified `decoder`.

This version performs allocation-less M-gaussian message updates. For the given variable node,
it computes mixture messages for all outgoing edges, then applies the forward-backward recursion
to update outgoing messages to all neighboring check nodes.

After computing the new messages, their means and variances are written back to the corresponding
check node message slots in the Tanner graph for use in the next iteration.

Arguments:
- `tg::TannerGraph`: The Tanner graph containing variable and check nodes.
- `vn_idx::Int64`: Index of the variable node to update.
- `decoder::LDLCDecoder`: The LDLC decoder with preallocated buffer structures.

Side Effects:
- Mutates internal message fields of neighboring check nodes for variable node `vn_idx`.
"""
function variable_node_messages_M_gaussian_allocationless!(tg::TannerGraph, vn_idx::Int64, decoder::LDLCDecoder)
    var_node = tg.var_nodes[vn_idx]
    d = length(var_node.neighbours)

    if d == 1
        cn_idx, _ = var_node.neighbours[1]
        idx = var_node.pos_in_check_neighbour[1]
        cn = tg.check_nodes[cn_idx]

        cn.messages[idx].mean = 1.0 * var_node.message.mean
        cn.messages[idx].var = 1.0 * var_node.message.var

    else
        for j = 1:d
            @views m_nearest!(decoder.allocs[d].mixtures[j], var_node.messages[j], var_node.message.mean, decoder.M)
        end

        forward_backward_recursion!(decoder, var_node.message, d)

        for j = 1:d
            cn_idx, _ = var_node.neighbours[j]
            idx = var_node.pos_in_check_neighbour[j]
            cn = tg.check_nodes[cn_idx]

            cn.messages[idx].mean = 1.0 * decoder.allocs[d].outputs[j].mean
            cn.messages[idx].var = 1.0 * decoder.allocs[d].outputs[j].var
        end
    end

end

"""
    forward_backward_recursion!(decoder, g, d)

Forward-backward recursion for M-Gaussian variable node update.
"""
function forward_backward_recursion!(decoder::LDLCDecoder, g::gaussian_log_weight, d::Int64)
    forward = decoder.allocs[d].forward
    backward = decoder.allocs[d].backward
    mixtures = decoder.allocs[d].mixtures
    combined = decoder.allocs[d].combined
    outputs = decoder.allocs[d].outputs
    ws_alloc = decoder.allocs[d].ws_alloc

    forward[1] .= mixtures[1]
    for j = 2:d
        prod!(forward[j], forward[j-1], mixtures[j])
    end

    backward[d] .= mixtures[d]
    for j = (d-1):-1:1
        prod!(backward[j], backward[j+1], mixtures[j])
    end

    for j = 1:d
        if j == 1
            prod!(combined, backward[j+1], g)
            moment_matching!(outputs[j], combined, ws_alloc)
        elseif  j == d
            prod!(combined, forward[j-1], g)
            moment_matching!(outputs[j], combined, ws_alloc)
        else
            prod!(combined, forward[j-1], backward[j+1])
            prod!(combined, g)
            moment_matching!(outputs[j], combined, ws_alloc)
        end
    end
end



function check_node_iterations!(dec::LDLCDecoder)
    for i = 1:dec.tg.nc
        check_node_messages!(dec.tg, i)
    end
end

"""
    run_decoder!(dec, message, σ, max_iter)

Run belief propagation decoder for `max_iter` iterations,
using signal `message` and noise standard deviation `σ`.
Returns posterior estimates.
"""
function run_decoder_parallel!(dec::LDLCDecoder, message::Vector{Float64}, σ::Float64, max_iter::Int64)
    initialize_messages!(dec.tg, message, σ)

    for i = 1:max_iter
        check_node_iterations!(dec)
        variable_node_iterations!(dec)
    end

    decision_step_nearest!(dec.tg)

    return dec.tg.bp_result
end
