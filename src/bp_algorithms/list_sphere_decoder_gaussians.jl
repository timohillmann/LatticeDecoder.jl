# Implementation of the simplied List Sphere Decoding algorithm proposed by Wang & Mow.
# Ref: X. Wang & W. Mow, "Efficient Decoder Design for Low-Density Lattice Codes From the Lattice Viewpoint", IEEE Open J. Commun. Soc. 4 1839-1854 (2023).
#
include("lsd_utils.jl")
# Parameters for the List Sphere Decoding algorithm
const W_MIN = 0.9
const EPSILON = 1e-10

"""
    lsd_variable_node_messages!(tg::TannerGraph, vn_idx::Int64)

Updates the messages of a variable node `vn_idx` in the Tanner graph `tg` using the List Sphere Decoding algorithm.
"""
function lsd_variable_node_messages!(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    for j = 1:length(vn.neighbours)
        cn_idx, _ = vn.neighbours[j]
        idx = vn.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]
        _lsd_variable_node_message!(cn.messages[idx], vn, j)
    end
end


"""
    _lsd_variable_node_message!(cn_message::gaussian, vn::VariableNode, nb_idx::Int)

Updates the message of a variable node `vn` for a specific neighbour `nb_idx` using the List Sphere Decoding algorithm.
"""
function _lsd_variable_node_message!(cn_message::gaussian, vn::VariableNode, nb_idx::Int)
    msg_vector = _collect_msg_vector(vn, nb_idx)
    lsd_inputs = ListSphereDecodingInput(msg_vector)
    L, D = simplified_lsd(lsd_inputs)
    if length(D) == 0
        println("Found length 0")
        return
    end
    candidate_gaussians = _calculate_candidate_gaussians(lsd_inputs, L, D, msg_vector)
    moment_matching!(cn_message, candidate_gaussians)
end


"""
    _collect_msg_vector(vn::VariableNode, j::Int64)

Collects the messages of a variable node `vn` for a specific neighbour `j` which will not be included in the message.
This vector is used to construct the input for the List Sphere Decoding algorithm.
"""
function _collect_msg_vector(vn::VariableNode, j::Int64)
    msg_vector = Vector{gaussian}()
    for i = 1:length(vn.messages)
        if i != j
            push!(msg_vector, vn.messages[i])
        end
    end
    push!(msg_vector, vn.message)
    return msg_vector
end


"""
    ListSphereDecodingInput(msg_vector::Vector{gaussian})

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
function ListSphereDecodingInput(msg_vector::Vector{gaussian})

    # Init storage vectors
    t_vector = zeros(Float64, length(msg_vector))
    g_vector = zeros(Float64, length(msg_vector))
    p_vector = zeros(Float64, length(msg_vector))

    # Init float values
    Vinv = 0.0
    β = 1.0


    for i = 1:(length(msg_vector))
        msg = msg_vector[i]
        Vinv += 1 / msg.var
        t_vector[i] = sign(msg.period) / sqrt(msg.var)
        g_vector[i] = sqrt(msg.var * msg.period^2)
        p_vector[i] = msg.mean * msg.period
        # β = abs(msg.period) < W_MIN ? max(β, 1 / sqrt(msg.var * msg.period^2)) : β  # Wang & Mow: Eq. (44)
        β = max(β, 1 / sqrt(msg.var * msg.period^2))
    end

    # overwrite the last element of p_vector with the mean of the last message
    p_vector[end] = msg_vector[end].mean
    u_d = msg_vector[end].mean / msg_vector[end].var / Vinv

    # Normalize t_vector
    t_vector *= 1 / sqrt(Vinv)

    # Calculate f_vector and R_vector based on precompyted vectors
    f_vector = _calculate_f_vector(t_vector)
    R_vector = _calculate_R_square_diag(g_vector, f_vector)

    # initialize LSD input
    return ListSphereDecodingInput(f_vector, g_vector, p_vector, t_vector, R_vector, 1 / Vinv, β, u_d)
end


function _calculate_candidate_gaussians(inputs::ListSphereDecodingInput, L::Vector, D::Vector{Float64}, msg_vector::Vector{gaussian})
    candidate_gaussians = Vector{gaussian}(undef, length(D))
    for i = 1:length(L)
        mean = 0.0
        var = inputs.Var
        weight = exp(-1 / 2 * D[i])
        log_weight = -1 / 2 * D[i]

        for j = 1:length(L[i])
            msg = msg_vector[j]
            mean += (msg.mean + L[i][j] / msg.period) / msg.var
        end
        mean *= var
        candidate_gaussians[i] = gaussian(mean, var, weight)
    end
    return candidate_gaussians
end