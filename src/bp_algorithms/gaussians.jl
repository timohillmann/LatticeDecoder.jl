const MIN_VAR::Float64 = 5e-6

abstract type Gaussian end

"""
    gaussian

A struct representing a (periodic) Gaussian distribution with mean `mean`, variance `var`, weight `weight`,
and optional periodic extension with period `period`.
"""
mutable struct gaussian <: Gaussian
    mean::Float64
    var::Float64
    weight::Float64
    period::Float64
    gaussian(mean, var, weight, period) =
        new(mean, max(var, MIN_VAR), weight, period)
end

gaussian(mean::Float64, var::Float64, weight::Float64) =
    gaussian(mean, var, weight, 0.0)

gaussian(mean::Float64, var::Float64) = gaussian(mean, var, 1.0, 0.0)

Base.zero(gaussian) = 0.0

Base.copy(g::gaussian) = gaussian(g.mean, g.var, g.weight, g.period)

function Base.isapprox(g1::gaussian, g2::gaussian)
    flag = true
    if !(g1.mean ≈ g2.mean)
        flag = false
    elseif !(g1.var ≈ g2.var)
        flag = false
    elseif !(g1.weight ≈ g2.weight)
        flag = false
    end
    return flag
end

struct TwoGaussianAlloc
    gL::gaussian
    gR::gaussian
end

mutable struct FourGaussianAlloc
    gL::gaussian
    gR::gaussian
    g1::gaussian
    g2::gaussian
end


"""
    nearest(g::gaussian, y::Float64, h::Float64, e::Float64=1.0)

Compute the two nearest Gaussian distributions `gL` and `gR` to a given
value `y`, based on a reference distribution `g`. An optional arugment `e`
describse the maximum distance allowed between the mean of a `gaussian`
distribution and the target value `y` for it to be considered a candidate for
the nearest distribution.

# Returns

A tuple `(gL, gR)` of `gaussian` distributions, where `gL` and `gR` are the two
nearest `gaussian` distributions to `y`.
"""
function nearest(g::gaussian, y::Float64, h::Float64, e::Float64=1.0)
    m = g.mean
    rhs = -(m - y) * h
    b1 = floor(rhs)
    b2 = b1 + 1

    # The left and right Gaussian N_{L,i}, N_{R,i} in Liu eq 19
    gL = gaussian(m - (b1 / h), g.var, g.weight)
    gR = gaussian(m - (b2 / h), g.var, g.weight)

    # gL and gR are preprocessed as in Liu, eqs. 22-24 
    if ((y - e) < gL.mean < (y + e)) && !((y - e) < gR.mean < (y + e))
        gR = gL
    elseif ((y - e) < gR.mean < (y + e)) && !((y - e) < gL.mean < (y + e))
        gL = gR
    end
    return (gL, gR)
end


function nearest!(g1::gaussian, g2::gaussian, g::gaussian, y::Float64, e::Float64=1.0)
    h = g.period
    m = g.mean
    rhs = (m - y) * h
    b1 = floor(rhs)
    b2 = b1 + 1

    @assert g.weight > 0.0

    left_mean = m - (b1 / h)
    right_mean = m - (b2 / h)

    g1.mean = left_mean
    g2.mean = right_mean
    g1.var = g.var
    g2.var = g.var
    g1.weight = g.weight
    g2.weight = g.weight

    if ((y - e) < left_mean < (y + e)) && !((y - e) < right_mean < (y + e))
        # right gaussian is the same as left
        g2.mean = left_mean

    elseif ((y - e) < right_mean < (y + e)) && !((y - e) < left_mean < (y + e))
        g1.mean = right_mean
    end

    if left_mean > right_mean
        g1.mean, g2.mean = g2.mean, g1.mean
    end

end

"""
    sum(g1::gaussian, g2::gaussian)

Compute the sum of two Gaussian distributions `g1` and `g2`. The resulting
distribution will have mean and variance given by:

    m = m1 * w1 + m2 * w2
    Δ = w1 * (Δ1 + m1^2) + w2 * (Δ2 + m2^2) - m^2

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively, and `w1` and `w2` are weights that determine the contribution
of each distribution.
"""
function Base.sum(g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    if isapprox(g2.weight, 0.0) && !isapprox(g1.weight, 0.0)
        return gaussian(g1.mean, g1.var, 1.0)
    elseif isapprox(g1.weight, 0.0) && !isapprox(g2.weight, 0.0)
        return gaussian(g2.mean, g2.var, 1.0)
    elseif isapprox(g2.weight, 0.0) && isapprox(g1.weight, 0.0)
        if g1.weight > g2.weight
            return gaussian(g1.mean, g1.var)
        else
            return gaussian(g2.mean, g2.var)
        end
    else
        w1 = g1.weight / (g1.weight + g2.weight)
        w2 = g2.weight / (g1.weight + g2.weight)
        m = m1 * w1 + m2 * w2
        Δ = w1 * (Δ1 + m1^2) + w2 * (Δ2 + m2^2) - m^2
        return gaussian(m, Δ)
    end
end

# Unused functions
function Base.sum!(g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    w1 = Δ1 / (Δ1 + Δ2)
    w2 = Δ2 / (Δ1 + Δ2)
    m = m1 * w1 + m2 * w2
    Δ = w1 * (Δ1 + m1^2) + w2 * (Δ2 + m2^2) - m^2

    g1.mean = m
    g1.var = max(Δ, MIN_VAR)
    g1.weight = 1.0
end

function sum!(g_out::gaussian, g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    if isapprox(g2.weight, 0.0) && !isapprox(g1.weight, 0.0)
        g_out.mean = 1.0 * g1.mean
        g_out.var = max(1.0 * g1.var, MIN_VAR)
        g_out.weight = 1.0
        # return gaussian(g1.mean, g1.var, 1.0)
    elseif isapprox(g1.weight, 0.0) && !isapprox(g2.weight, 0.0)
        g_out.mean = 1.0 * g2.mean
        g_out.var = max(1.0 * g2.var, MIN_VAR)
        g_out.weight = 1.0
        # return gaussian(g2.mean, g2.var, 1.0)
    elseif isapprox(g2.weight, 0.0) && isapprox(g1.weight, 0.0)
        if g1.weight > g2.weight
            # return gaussian(g1.mean, g1.var)
            g_out.mean = 1.0 * g1.mean
            g_out.var = max(1.0 * g1.var, MIN_VAR)
            g_out.weight = 1.0
        else
            # return gaussian(g2.mean, g2.var)
            g_out.mean = 1.0 * g2.mean
            g_out.var = max(1.0 * g2.var, MIN_VAR)
            g_out.weight = 1.0
        end
    else
        w1 = g1.weight / (g1.weight + g2.weight)
        w2 = g2.weight / (g1.weight + g2.weight)
        m = m1 * w1 + m2 * w2
        Δ = w1 * (Δ1 + m1^2) + w2 * (Δ2 + m2^2) - m^2
        g_out.mean = m
        g_out.var = max(Δ, MIN_VAR)
        g_out.weight = 1.0
        # return gaussian(m, Δ)
    end
end


function Base.prod(g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    Δ = 1 / (1 / Δ1 + 1 / Δ2)
    m = Δ * (m1 / Δ1 + m2 / Δ2)
    c = exp(-(m1 - m2)^2 / (2 * (Δ1 + Δ2))) / (sqrt(2 * pi * (Δ1 + Δ2)))
    return gaussian(m, Δ, c * g1.weight * g2.weight)
end

"""
    prod(g1::gaussian, g2::gaussian)

Compute the product of two Gaussian distributions `g1` and `g2`. The resulting
distribution will have mean and variance given by:

    m = Δ * (m1 / Δ1 + m2 / Δ2)
    Δ = 1 / (1 / Δ1 + 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting weight `c` is also computed and the results are
modified in place.
"""
function Base.prod!(g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    @assert g1.weight > 0.0
    @assert g2.weight > 0.0

    Δ = 1 / (1 / Δ1 + 1 / Δ2)
    m = Δ * (m1 / Δ1 + m2 / Δ2)
    c = exp(-(m1 - m2)^2 / (2 * (Δ1 + Δ2))) / (sqrt(2 * pi * (Δ1 + Δ2)))
    # log_c = -(m1 - m2)^2 / (2 * (Δ1 + Δ2)) - log((sqrt(2 * pi * (Δ1 + Δ2))))

    g1.mean = m
    g1.var = Δ
    # g1.weight = exp(log_c) * g1.weight * g2.weight
    g1.weight = c * g1.weight * g2.weight

    @assert g1.weight > 0.0 "$(m1), $(m2)"
end


"""
    divide(g1::gaussian, g2::gaussian)

Compute the division of two Gaussian distributions `g1` and `g2`. The resulting
distribution will have mean and variance given by:

    m = Δ * (m1 / Δ1 - m2 / Δ2)
    Δ = 1 / (1 / Δ1 - 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting distribution has the correct weight `β`. 
The expression is obtained from http://davmre.github.io/blog/statistics/2015/03/27/gaussian_quotient

The function returns a new `gaussian` distribution with weight `β`.

"""
function divide(g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var
    c1 = g1.weight
    c2 = g2.weight

    @assert Δ2 > Δ1

    Δ = 1 / (1 / Δ1 - 1 / Δ2)
    m = Δ * (m1 / Δ1 - m2 / Δ2)
    # β = (sqrt(2 * pi * (Δ2 - Δ1))) * exp(-(m1 - m2)^2 / (2 * (Δ2 - Δ1))) * (Δ2 / (Δ2 - Δ1)) * c1 / c2
    log_β = (m1 - m2)^2 / (2 * (Δ2 - Δ1)) + 0.5 * log(2 * pi) + log(Δ2) - 0.5 * log(Δ2 - Δ1) + log(c1) - log(c2)

    #println(log_β)

    return gaussian(m, Δ, exp(log_β))
end


"""
    divide!(g1::gaussian, g2::gaussian)

Divide the Gaussian distribution `g1` by `g2`. The resulting distribution will
have mean and variance given by:

    m = Δ * (m1 / Δ1 - m2 / Δ2)
    Δ = 1 / (1 / Δ1 - 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting distribution has the correct weight `β`.

The results are modified in place.
"""
function divide!(g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var
    c1 = g1.weight
    c2 = g2.weight

    Δ = 1 / (1 / Δ1 - 1 / Δ2)
    m = Δ * (m1 / Δ1 - m2 / Δ2)
    # β = (sqrt(2 * pi * (Δ2 - Δ1))) * exp(-(m1 - m2)^2 / (2 * (Δ2 - Δ1))) * (Δ2 / (Δ2 - Δ1)) * c1 / c2
    log_β = (m1 - m2)^2 / (2 * (Δ2 - Δ1)) + 0.5 * log(2 * pi) + log(Δ2) - 0.5 * log(Δ2 - Δ1) + log(c1) - log(c2)
    g1.mean = m
    g1.var = Δ
    g1.weight = exp(log_β)
end


"""
    divide!(g_out::gaussian, g1::gaussian, g2::gaussian)

Divide the Gaussian distribution `g1` by `g2`. The resulting distribution will
have mean and variance given by:

    m = Δ * (m1 / Δ1 - m2 / Δ2)
    Δ = 1 / (1 / Δ1 - 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,

The results are modified in place and written into `g_out`.

"""
function divide!(g_out::gaussian, g1::gaussian, g2::gaussian)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var
    c1 = g1.weight
    c2 = g2.weight

    Δ = 1 / (1 / Δ1 - 1 / Δ2)
    m = Δ * (m1 / Δ1 - m2 / Δ2)
    #β = (sqrt(2 * pi * (Δ2 - Δ1))) * exp(-(m1 - m2)^2 / (2 * (Δ2 - Δ1))) * (Δ2 / (Δ2 - Δ1)) * c1 / c2
    log_β = (m1 - m2)^2 / (2 * (Δ2 - Δ1)) + 0.5 * log(2 * pi) + log(Δ2) - 0.5 * log(Δ2 - Δ1) + log(c1) - log(c2)

    g_out.mean = m
    g_out.var = Δ
    g_out.weight = exp(log_β)
end


function moment_matching(gs::AbstractVector{gaussian})
    ws = Vector{Float64}(undef, length(gs))
    for i in eachindex(ws)
        ws[i] = gs[i].weight
    end

    ws = ws / sum(ws)

    m = sum([g.mean * w for (g, w) in zip(gs, ws)])

    Δ = sum([w * (g.var + g.mean^2) for (g, w) in zip(gs, ws)]) - m^2

    return gaussian(m, max(Δ, MIN_VAR))
end

function moment_matching!(g_out::gaussian, gs::AbstractVector{gaussian})
    ws = Vector{Float64}(undef, length(gs))
    for i in eachindex(ws)
        ws[i] = gs[i].weight
    end

    ws = ws / sum(ws)

    m = sum([g.mean * w for (g, w) in zip(gs, ws)])

    Δ = sum([w * (g.var + g.mean^2) for (g, w) in zip(gs, ws)]) - m^2

    g_out.mean = m
    g_out.var = max(Δ, MIN_VAR)
    g_out.weight = 1.0
end




# g1 = gaussian(0.1, 0.1, 0.1)
# g2 = gaussian(-0.2, 0.2, 0.2)
# g3 = gaussian(-0.3, 0.3, 0.3)

# g12 = prod(g1, g2)
# g23 = prod(g2, g3)
# g13 = prod(g1, g3)

# divide(g12, g1)
# g = gaussian(0.0, 0.5, 0.5)

# isapprox(divide(g12, g1), g2)
# isapprox(divide(g12, g2), g1)
# isapprox(divide(g23, g2), g3)
# isapprox(divide(g23, g3), g2)
# isapprox(divide(g13, g1), g3)
# isapprox(divide(g13, g3), g1)


# divide!(g, g12, g1)
# @assert g2 ≈ g

# divide!(g, g12, g2)
# @assert g1 ≈ g

# divide!(g, g23, g2)
# @assert g3 ≈ g

# divide!(g, g23, g3)
# @assert g2 ≈ g

# divide!(g, g13, g3)
# @assert g1 ≈ g

# divide!(g, g13, g1)
# @assert g3 ≈ g

