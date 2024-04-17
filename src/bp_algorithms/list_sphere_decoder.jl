

const W_MIN = 0.9


function _lsd_variable_node_messages!(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    for nb_idx = 1:length(vn.neighbours)
        _lsd_variable_node_message!(tg, vn_idx, nb_idx)
    end
end

function _lsd_variable_node_message!(tg::TannerGraph, vn_idx::Int, nb_idx::Int)
    msg_vector = _collect_msg_vector(tg, vn_idx, nb_idx)
    lsd_inputs = ListSphereDecodingInput(msg_vector)
    L, D = simplified_lsd(lsd_inputs)
    candidate_gaussians = _calculate_candidate_gaussians(lsd_inputs, L, D, msg_vector)
    moment_matching!(cn.messages[idx], candidate_gaussians)
end



function _collect_msg_vector(vn::VariableNode, nb_idx::Int64)
    cn_idx, _ = vn.neighbours[nb_idx]
    idx = vn.pos_in_check_neighbour[nb_idx]

    # create vector of length d of type gaussian for the messages
    msg_vector = Vector{gaussian}()
    for i = 1:length(vn.messages)
        if i != nb_idx
            push!(msg_vector, vn.messages[i])
        end
    end
    push!(msg_vector, vn.message)
    # msg_vector[end].period = 1.0
    return msg_vector
end


mutable struct ListSphereDecodingInput
    f_vector::Vector{Float64}
    g_vector::Vector{Float64}
    p_vector::Vector{Float64}
    t_vector::Vector{Float64}
    R_vector::Vector{Float64}
    Var::Float64
    β::Float64
    u_d::Float64
end


"""
    ListSphereDecodingInput(msg_vector::Vector{Gaussian})

Constructs the input for the List Sphere Decoding algorithm from a vector of Gaussian messages.

The input consists of the following vectors:
- `f_vector::Vector{Float64}`: The f vector.
- `g_vector::Vector{Float64}`: The g vector.
- `p_vector::Vector{Float64}`: The p vector.
- `t_vector::Vector{Float64}`: The t vector.
- `R_vector::Vector{Float64}`: The R vector.
- `Var::Float64`: The variance of the message vector.
- `β::Float64`: The β parameter.
- `u_d::Float64`: The u_d parameter.
"""
function ListSphereDecodingInput(msg_vector::Vector{gaussian})
    t_vector = zeros(Float64, length(msg_vector))
    g_vector = zeros(Float64, length(msg_vector))
    p_vector = zeros(Float64, length(msg_vector))
    Vinv = 0.0
    β = 1.0
    for i = 1:(length(msg_vector))
        msg = msg_vector[i]
        Vinv += 1 / msg.var
        t_vector[i] = sign(msg.period) / sqrt(msg.var)
        g_vector[i] = sqrt(msg.var * msg.period^2)
        p_vector[i] = msg.mean * msg.period
        β = abs(msg.period) < W_MIN ? max(β, 1 / sqrt(msg.var * msg.period^2)) : β
    end

    # overwrite the last element of p_vector with the mean of the last message
    p_vector[end] = msg_vector[end].mean
    u_d = msg_vector[end].mean / msg_vector[end].var / Vinv

    t_vector *= 1 / sqrt(Vinv)
    f_vector = _calculate_f_vector(t_vector)

    R_vector = _calculate_R_square_diag(g_vector, f_vector)

    ListSphereDecodingInput(f_vector, g_vector, p_vector, t_vector, R_vector, 1 / Vinv, β, u_d)
end


"""
    _calculate_f_vector(t_vector::Vector{Float64})

f_i = 1 - Σ_{l = 1}^{i} t_l^2
"""
function _calculate_f_vector(t_vector::Vector{Float64})
    f_vector = zeros(Float64, length(t_vector))
    f_vector[1] = 1 - t_vector[1]^2
    for i = 2:length(t_vector)
        f_vector[i] = f_vector[i-1] - t_vector[i]^2
    end
    return f_vector
end


"""
    _calculate_R_square_diag(_calculate_R_diag(g_vector::Vector{Float64}, f_vector::Vector{Float64})


R_ii    = 1 / sqrt(h_i^2 v_i) * sqrt(1 - Σ_{l = 1}^{i} t_l^2) / sqrt(1 - Σ_{l = 1}^{i - 1} t_l^2)
        = 1 / g_vector[i] * sqrt(f_vector[i]) / sqrt(f_vector[i-1])

"""
function _calculate_R_square_diag(g_vector::Vector{Float64}, f_vector::Vector{Float64})
    R_square_diag = zeros(Float64, length(g_vector))
    R_square_diag[1] = 1 / g_vector[1]^2 * f_vector[1]
    for i = 2:(length(g_vector)-1)
        R_square_diag[i] = 1 / g_vector[i]^2 * f_vector[i] / f_vector[i-1]
    end
    return R_square_diag
end


"""
    simplified_lsd(inputs::AbstractNode)

    Simplified version of the List Sphere Decoding algorithm.

"""
function simplified_lsd(inputs::ListSphereDecodingInput)


    # Point to inputs
    p = inputs.p_vector
    t = inputs.t_vector
    g = inputs.g_vector
    f = inputs.f_vector
    R_sq = inputs.R_vector
    # β = inputs.β
    d = length(p)

    k = d - 1
    dist = zeros(Float64, d)
    L = Vector{Vector{Int16}}()
    D = Vector{Float64}()

    z = zeros(Float64, d)
    s = zeros(Float64, d)
    gamma = zeros(Float64, d)
    u = zeros(Float64, d)
    u[d] = inputs.u_d

    # Steps 2-5:
    gamma[k] = -p[k] + t[k] * g[k] * u[k+1] / f[k]
    z[k] = round(gamma[k])
    s[k] = sign(gamma[k] - z[k])
    dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]

    while k <= (d - 1)
        if dist[k] < (inputs.β)^2
            if k == 1
                push!(L, round.(Int16, copy(z)))
                push!(D, copy(dist[k]))
                if length(D) == 1
                    update_beta!(inputs, D[1])
                end

                # Steps 10-12
                z[k] += s[k]
                s[k] = -s[k] - sign(s[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]

            else
                u[k] = t[k] * (z[k] + p[k]) / g[k] + u[k+1]
                k -= 1

                # Repeat Steps 2-5
                gamma[k] = -p[k] + t[k] * g[k] * u[k+1] / f[k]
                z[k] = round(gamma[k])
                s[k] = sign(gamma[k] - z[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
            end # if k == 1

        else
            if k == (d - 1)
                return L, D
            else
                k += 1

                # Repeat Steps 10-12
                z[k] += s[k]
                s[k] = -s[k] - sign(s[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
            end # if k == (d-1)

        end # if dist
    end # while
end


function update_beta!(inputs::ListSphereDecodingInput, DB::Float64, ϵ=1e-10::Float64)
    lsd_inputs.β = min(inputs.β, sqrt(DB^2 + 2 * log(1 / ϵ)))
end
