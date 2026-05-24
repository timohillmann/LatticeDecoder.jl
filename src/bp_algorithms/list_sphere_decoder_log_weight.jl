# Implementation of the simplied List Sphere Decoding algorithm proposed by Wang & Mow.
# Ref: X. Wang & W. Mow, "Efficient Decoder Design for Low-Density Lattice Codes From the Lattice Viewpoint", IEEE Open J. Commun. Soc. 4 1839-1854 (2023).
#
include("lsd_utils.jl")

"""
    lsd_variable_node_messages_reference!(tg::TannerGraph, vn_idx::Int64)

Allocating reference implementation of the List Sphere Decoding variable-node update.
"""
function lsd_variable_node_messages_reference!(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    for j = 1:length(vn.neighbours)
        cn_idx, _ = vn.neighbours[j]
        idx = vn.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]
        _lsd_variable_node_message!(cn.messages[idx], vn, j; beta=tg.lsd_beta, w_min=tg.lsd_w_min)
    end
end


"""
    _lsd_variable_node_message!(cn_message::gaussian_log_weight, vn::VariableNode, nb_idx::Int)

Updates the message of a variable node `vn` for a specific neighbour `nb_idx` using the List Sphere Decoding algorithm.
"""
function _lsd_variable_node_message!(cn_message::gaussian_log_weight, vn::VariableNode, nb_idx::Int; beta::Float64=LatticeDecoder.LSD_DEFAULT_BETA, w_min::Float64=LatticeDecoder.LSD_W_MIN)
    


    msg_vector = _collect_msg_vector(vn, nb_idx)
    if length(msg_vector) == 1
        cn_message.mean = msg_vector[1].mean
        cn_message.var = vn.message.var  # LatticeDecoder.MIN_VAR  # or msg_vector[1].var
        cn_message.period = msg_vector[1].period
        return 1
    end
    lsd_inputs = ListSphereDecodingInput(msg_vector; beta=beta, w_min=w_min)
    L, D = simplified_lsd(lsd_inputs)
    if length(D) == 0
        # print("Nothing found.")
        # printstyled("msg_v: $(msg_vector), cn_msg: $(cn_message)\n", color=:red)
        return 0

    end
    candidate_gaussians = _calculate_candidate_gaussians(lsd_inputs, L, D, msg_vector)
    moment_matching!(cn_message, candidate_gaussians)
    return length(D)
end


"""
    _lsd_variable_node_decision_reference!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)

Allocating reference implementation of the List Sphere Decoding decision step.
"""
function _lsd_variable_node_decision_reference!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    msg_vector = _collect_msg_vector(vn)

    if length(msg_vector) == 2  # handle weird case
        for msg in msg_vector
            if msg.var ≈ LatticeDecoder.MIN_VAR # this will dominate
                bp_result[vn_idx] = msg.mean
                return 
            end
        end
    end

    lsd_inputs = ListSphereDecodingInput(msg_vector; beta=tg.lsd_beta, w_min=tg.lsd_w_min)
    L, D = simplified_lsd(lsd_inputs)
    candidate_gaussians = _calculate_candidate_gaussians(lsd_inputs, L, D, msg_vector)
    if length(D) == 0
        print("Nothing found. vn.message.mean = $(vn.message.mean)")
        print(msg_vector)
        print("Number messages: $(length(msg_vector))")
        # vn.message.mean = 0.0 # no candidate found
    else
        moment_matching!(vn.message, candidate_gaussians)
    end
    bp_result[vn_idx] = vn.message.mean
end

"""
    _collect_msg_vector(vn::VariableNode, j::Int64)

Collects the messages of a variable node `vn` for a specific neighbour `j` which will not be included in the message.
This vector is used to construct the input for the List Sphere Decoding algorithm.
"""
function _collect_msg_vector(vn::VariableNode, j::Int64)
    msg_vector = Vector{gaussian_log_weight}()
    for i = 1:length(vn.messages)
        if i != j
            push!(msg_vector, vn.messages[i])
        end
    end
    push!(msg_vector, vn.message)
    return msg_vector
end


"""
    _collect_msg_vector(vn::VariableNode)

Collects the message of a variable node `vn` for all neighbouring check nodes.
"""
function _collect_msg_vector(vn::VariableNode)
    msg_vector = Vector{gaussian_log_weight}()
    for i = 1:length(vn.messages)
        push!(msg_vector, vn.messages[i])
    end
    push!(msg_vector, vn.message)
    return msg_vector
end

_collect_msg_vector(vn::VariableNode, j::Nothing) = _collect_msg_vector(vn)


"""
    ListSphereDecodingInput(msg_vector::Vector{gaussian_log_weight})

Constructs the input for the List Sphere Decoding algorithm from a vector of Gaussian messages.

The input consists of the following vectors:
- `f_vector::Vector{Float64}`: The f vector, f[i] = 1 - Σ_{l = 1}^{i} t_l^2. Wang & Mow: Eq. (48)
- `g_vector::Vector{Float64}`: The g vector, g[i] = sqrt(var[i] * period[i]^2). Wang & Mow: Eq. (49)
- `p_vector::Vector{Float64}`: The p vector, It elements are given by the product of the mean and the period of the message. Wang & Mow: before Eq. (46)
- `t_vector::Vector{Float64}`: The t vector, t[i] = sign(period[i]) * sqrt(V / var[i]). Wang & Mow: after Eq. (40)
- `R_vector::Vector{Float64}`: The squared R vector, R[i] = 1 / g[i]^2 * f[i]) / sqrt(f[i-1]. Wang & Mow: Eq. (42)
- `Var::Float64`: The variance of the message vector.
- `β::Float64`: The β parameter, the search radius.
- `u_d::Float64`: The u_d parameter, u_d = sqrt(Var) * mean[d] / var[d]. Wang & Mow: after Eq. (51)
"""
function ListSphereDecodingInput(msg_vector::Vector{gaussian_log_weight}; beta::Float64=LatticeDecoder.LSD_DEFAULT_BETA, w_min::Float64=LatticeDecoder.LSD_W_MIN)
    beta > 0 || throw(ArgumentError("LSD beta must be positive."))
    w_min >= 0 || throw(ArgumentError("LSD w_min must be nonnegative."))

    # Init storage vectors
    t_vector = zeros(Float64, length(msg_vector))
    g_vector = zeros(Float64, length(msg_vector))
    p_vector = zeros(Float64, length(msg_vector))

    # Init float values
    Vinv = 0.0
    β = beta


    for i = 1:(length(msg_vector))
        msg = msg_vector[i]
        Vinv += 1 / msg.var
        t_vector[i] = sign(msg.period) / sqrt(msg.var)
        g_vector[i] = sqrt(msg.var * msg.period^2)
        p_vector[i] = msg.mean * msg.period
        # print("Var & Period ", msg.var, " ", msg.period, " ")
        β = abs(msg.period) < w_min ? max(β, 1 / sqrt(msg.var * msg.period^2)) : β  # Wang & Mow: Eq. (44)
        # β = max(β, 1 / sqrt(msg.var * msg.period^2))
    end
    # overwrite the last element of p_vector with the mean of the last message
    u_d = msg_vector[end].mean / msg_vector[end].var / sqrt(Vinv)

    # Normalize t_vector
    t_vector *= 1 / sqrt(Vinv)

    # Calculate f_vector and R_vector based on precompyted vectors
    f_vector = _calculate_f_vector(t_vector)
    R_vector = _calculate_R_square_diag(g_vector, f_vector)

    # initialize LSD input
    return ListSphereDecodingInput(f_vector, g_vector, p_vector, t_vector, R_vector, 1 / Vinv, copy(β), copy(β), u_d)
end


function _calculate_candidate_gaussians(inputs::ListSphereDecodingInput, L::Vector, D::Vector{Float64}, msg_vector::Vector{gaussian_log_weight})
    candidate_gaussians = Vector{gaussian_log_weight}(undef, length(D))
    var = inputs.Var
    for i = 1:length(L)
        mean = 0.0
        log_weight = -1 / 2 * D[i]
        for j = 1:length(L[i])
            msg = msg_vector[j]
            mean += (msg.mean + L[i][j] / msg.period) / msg.var
        end
        mean *= var
        candidate_gaussians[i] = gaussian_log_weight(mean, copy(var), log_weight)
    end

    # rearrange ordering of gaussians based on distace
    # inds = sortperm(D)
    # candidate_gaussians = candidate_gaussians[inds]
    # if length(candidate_gaussians) > 2
    #     print("Shorteninng output.")
    #     candidate_gaussians = candidate_gaussians[1:2]
    # end
    return candidate_gaussians
end


"""
    _lsd_variable_node_decision_reference!(tg::TannerGraph, vn_idx::Int64)

Allocating reference implementation of the variable-node decision step.
"""
function _lsd_variable_node_decision_reference!(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    msg_vector = _collect_msg_vector(vn)

    # I need to handle the no checks case.
    if length(msg_vector) == 2
        for msg in msg_vector
            if msg.var ≈ LatticeDecoder.MIN_VAR # this will dominate
                tg.bp_result[vn_idx] = msg.mean
                return 
            end
        end
    end

    lsd_inputs = ListSphereDecodingInput(msg_vector; beta=tg.lsd_beta, w_min=tg.lsd_w_min)

    L, D = simplified_lsd(lsd_inputs)
    can_gaussian = _calculate_candidate_gaussians(lsd_inputs, L, D, msg_vector)

    if length(D) == 0
        print("Nothing found.")
    else
        moment_matching!(vn.message, can_gaussian)
    end
    # print("Candidate Gaussians: ", can_gaussian)
    tg.bp_result[vn_idx] = vn.message.mean
end

function decision_step_lsd_reference!(tg::TannerGraph)
    for vn_idx = 1:tg.nv
        _lsd_variable_node_decision_reference!(tg, vn_idx)
    end
    return nothing
end

function run_belief_propagation_lsd_reference!(
    tg::TannerGraph,
    message::Vector{Float64},
    σ::Float64,
    max_iter::Int64;
    search_interval::Float64=1.5,
)
    tg.search_interval = search_interval
    initialize_messages!(tg, message, σ)

    for _ = 1:max_iter
        check_node_iterations!(tg)
        for vn_idx = 1:tg.nv
            lsd_variable_node_messages_reference!(tg, vn_idx)
        end
    end

    decision_step_lsd_reference!(tg)
    return tg.bp_result
end
