# This file contains the List Sphere Decoding algorithm implementation.
# All of the functions here are independent of the messages being of type `gaussian` of `gaussian_log_weight`.

mutable struct ListSphereDecodingInput
    f_vector::Vector{Float64}
    g_vector::Vector{Float64}
    p_vector::Vector{Float64}
    t_vector::Vector{Float64}
    R_vector::Vector{Float64}
    Var::Float64
    β::Float64
    β1::Float64
    u_d::Float64
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

struct LSDSearchResult
    L::Vector{Vector{Int16}}
    D::Vector{Float64}
    status::Symbol
    visits::Int
end

_schnorr_euchner_step(delta::Float64) = delta < 0 ? -1.0 : 1.0

"""
    simplified_lsd_legacy(inputs::ListSphereDecodingInput)

Legacy List Sphere Decoding search with the historical fixed iteration cap.
"""
function simplified_lsd_legacy(inputs::ListSphereDecodingInput)
    # Point to inputs
    p = inputs.p_vector
    t = inputs.t_vector
    g = inputs.g_vector
    f = inputs.f_vector
    R_sq = inputs.R_vector
    d = length(p)

    # Initialize variables
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
    iter = 0
    while k <= (d - 1) && iter <= LatticeDecoder.MAX_ITER
        iter += 1
        if dist[k] <= (inputs.β)^2
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
                # printstyled("Returned after $(iter) iterations.\n", color=:blue)
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
    # printstyled("Returned after $(iter) iterations.\n", color=:red)
    return L, D
end

"""
    simplified_lsd_paper(inputs; max_visits=nothing)

Paper-faithful List Sphere Decoding search. This removes the legacy fixed
iteration cap and reports why the search stopped.
"""
function simplified_lsd_paper(
    inputs::ListSphereDecodingInput;
    max_visits::Union{Nothing,Int}=nothing,
)
    max_visits === nothing || max_visits >= 0 || throw(ArgumentError("max_visits must be nonnegative"))

    p = inputs.p_vector
    t = inputs.t_vector
    g = inputs.g_vector
    f = inputs.f_vector
    R_sq = inputs.R_vector
    d = length(p)
    d >= 2 || throw(ArgumentError("simplified_lsd requires at least two input messages"))

    k = d - 1
    dist = zeros(Float64, d)
    L = Vector{Vector{Int16}}()
    D = Vector{Float64}()

    z = zeros(Float64, d)
    s = zeros(Float64, d)
    gamma = zeros(Float64, d)
    u = zeros(Float64, d)
    u[d] = inputs.u_d

    beta = inputs.β
    beta1 = inputs.β1
    beta_sq = beta^2

    gamma[k] = -p[k] + t[k] * g[k] * u[k+1] / f[k]
    z[k] = round(gamma[k])
    s[k] = _schnorr_euchner_step(gamma[k] - z[k])
    dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]

    visits = 0
    while k <= (d - 1)
        if max_visits !== nothing && visits >= max_visits
            return LSDSearchResult(L, D, :budget_exhausted, visits)
        end
        visits += 1

        if dist[k] <= beta_sq
            if k == 1
                push!(L, round.(Int16, copy(z)))
                push!(D, copy(dist[k]))
                if length(D) == 1
                    beta = min(beta1, sqrt(D[1] - 2 * log(LatticeDecoder.EPSILON)))
                    beta_sq = beta^2
                end

                z[k] += s[k]
                s[k] = -s[k] - sign(s[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
            else
                u[k] = t[k] * (z[k] + p[k]) / g[k] + u[k+1]
                k -= 1

                gamma[k] = -p[k] + t[k] * g[k] * u[k+1] / f[k]
                z[k] = round(gamma[k])
                s[k] = _schnorr_euchner_step(gamma[k] - z[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
            end
        else
            if k == (d - 1)
                status = isempty(D) ? :empty : :ok
                return LSDSearchResult(L, D, status, visits)
            else
                k += 1

                z[k] += s[k]
                s[k] = -s[k] - sign(s[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
            end
        end
    end

    status = isempty(D) ? :empty : :ok
    return LSDSearchResult(L, D, status, visits)
end

"""
    simplified_lsd(inputs::ListSphereDecodingInput)

Default List Sphere Decoding search. Returns the same `(L, D)` tuple shape as
the legacy function, backed by the paper-faithful search.
"""
function simplified_lsd(inputs::ListSphereDecodingInput)
    result = simplified_lsd_paper(inputs)
    return result.L, result.D
end


"""
    update_beta!(inputs::ListSphereDecodingInput, DB::Float64, ϵ=LatticeDecoder.EPSILON::Float64)

Updates the β parameter of the List Sphere Decoding algorithm, see Wang & Mow: Eq. (45).
"""
function update_beta!(inputs::ListSphereDecodingInput, DB::Float64, ϵ=LatticeDecoder.EPSILON::Float64)
    inputs.β = min(inputs.β1, sqrt(DB - 2 * log(ϵ)))
end
