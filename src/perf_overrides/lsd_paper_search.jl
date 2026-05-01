# Paper-faithful List Sphere Decoding search for diagnostics and tests.
#
# This intentionally does not replace simplified_lsd. It mirrors Wang-Mow
# Algorithm 1 without the legacy fixed iteration cap, and reports why the
# search stopped.

struct LSDSearchResult
    L::Vector{Vector{Int16}}
    D::Vector{Float64}
    status::Symbol
    visits::Int
end

_paper_schnorr_euchner_step(delta::Float64) = delta < 0 ? -1.0 : 1.0

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
    d >= 2 || throw(ArgumentError("simplified_lsd_paper requires at least two input messages"))

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
    s[k] = _paper_schnorr_euchner_step(gamma[k] - z[k])
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
                    beta = min(beta1, sqrt(D[1] - 2 * log(EPSILON)))
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
                s[k] = _paper_schnorr_euchner_step(gamma[k] - z[k])
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
