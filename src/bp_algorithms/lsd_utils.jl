

struct ListSphereDecodingInput
    f_vector::Vector{Float64}
    g_vector::Vector{Float64}
    p_vector::Vector{Float64}
    t_vector::Vector{Float64}
    R_vector::Vector{Float64}
    Vinv::Float64
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
function ListSphereDecodingInput(msg_vector::Vector{Gaussian})
    t_vector = zeros(Float64, length(msg_vector))
    g_vector = zeros(Float64, length(msg_vector))
    p_vector = zeros(Float64, length(msg_vector))
    Vinv = 0.0
    β = 1.0
    for i = 1:(length(msg_vector))
        msg = msg_vector[i]
        Vinv += 1 / msg.var
        push!(t_vector, sign(msg.period) / sqrt(msg.var))
        push!(g_vector, sqrt(msg.var * msg.period^2))
        push!(p_vector, msg.mean * msg.period)
    end

    # overwrite the last element of p_vector with the mean of the last message
    p_vector[end] = msg_vector[end].mean
    u_d = msg_vector[end].mean / msg_vector[end].var / Vinv

    t_vector *= 1 / sqrt(Vinv)
    g_vector .*= abs.(h_vector)
    f_vector = _calculate_f_vector(t_vector)

    R_vector = _calculate_R_diag(g_vector, f_vector)

    new(f_vector, g_vector, p_vector, t_vector, R_vector, 1 / Vinv, β, u_d)
end


