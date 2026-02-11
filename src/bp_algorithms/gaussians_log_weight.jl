const MIN_VAR::Float64 = 1e-10

abstract type Gaussian end

"""
    gaussian_log_weigth

A struct representing a (periodic) Gaussian distribution with mean `mean`, variance `var`, weight `weight`,
and optional periodic extension with period `period`.
"""
mutable struct gaussian_log_weight <: Gaussian
    mean::Float64
    var::Float64
    log_weight::Float64
    period::Float64
    gaussian_log_weight(mean, var, log_weight, period) =
        new(mean, max(var, MIN_VAR), log_weight, period)
end

gaussian_log_weight(mean::Float64, var::Float64, log_weight::Float64) =
    gaussian_log_weight(mean, var, log_weight, 1.0)

gaussian_log_weight(mean::Float64, var::Float64) = gaussian_log_weight(mean, var, 0.0, 1.0)

Base.zero(gaussian_log_weight) = 0.0

Base.copy(g::gaussian_log_weight) = gaussian_log_weight(g.mean, g.var, g.log_weight, g.period)

function Base.isapprox(g1::gaussian_log_weight, g2::gaussian_log_weight; atol=1e-6, rtol=1e-6)
    return isapprox(g1.mean, g2.mean, atol=atol, rtol=rtol) &&
           isapprox(g1.var, g2.var, atol=atol, rtol=rtol) &&
           isapprox(g1.log_weight, g2.log_weight, atol=atol, rtol=rtol)
end

mutable struct FourGaussianLogAlloc
    gL::gaussian_log_weight
    gR::gaussian_log_weight
    g1::gaussian_log_weight
    g2::gaussian_log_weight

    FourGaussianLogAlloc(g::gaussian_log_weight) = new(copy(g), copy(g), copy(g), copy(g))
end


mutable struct SixGaussianLogAlloc
    gL::gaussian_log_weight
    gR::gaussian_log_weight
    g1::gaussian_log_weight
    g2::gaussian_log_weight
    gL_j::gaussian_log_weight
    gR_j::gaussian_log_weight

    SixGaussianLogAlloc(g::gaussian_log_weight) = new(copy(g), copy(g), copy(g), copy(g), copy(g), copy(g))

end


"""
    prod!(g1::gaussian, g2::gaussian)

Update the Gaussian distribution `g1` to be the product of `g1` and `g2`. The
resulting distribution will have mean and variance given by:

    m = Δ * (m1 / Δ1 + m2 / Δ2)
    Δ = 1 / (1 / Δ1 + 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting weight `c` is also computed and assigned to `g1`.
"""
function Base.prod!(g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    Δ = 1 / (1 / Δ1 + 1 / Δ2)
    m = Δ * (m1 / Δ1 + m2 / Δ2)
    c = -(m1 - m2)^2 / (2 * (Δ1 + Δ2)) - log((sqrt(2 * pi * (Δ1 + Δ2))))

    g1.mean = m
    g1.var = max(Δ, MIN_VAR)
    g1.log_weight = c + g1.log_weight + g2.log_weight
end

"""
    prod(g1::gaussian, g2::gaussian)

Update the Gaussian distribution `g1` to be the product of `g1` and `g2`. The
resulting distribution will have mean and variance given by:

    m = Δ * (m1 / Δ1 + m2 / Δ2)
    Δ = 1 / (1 / Δ1 + 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting weight `c` is also computed and assigned to `g1`.
"""
function Base.prod(g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    Δ = max(1 / (1 / Δ1 + 1 / Δ2), MIN_VAR)
    m = Δ * (m1 / Δ1 + m2 / Δ2)
    c = -(m1 - m2)^2 / (2 * (Δ1 + Δ2)) - log((sqrt(2 * pi * (Δ1 + Δ2))))
    return gaussian_log_weight(m, Δ, c + g1.log_weight + g2.log_weight)
end

"""
    prod(dest::gaussian, g1::gaussian, g2::gaussian)

Update the Gaussian distribution `g1` to be the product of `g1` and `g2`. The
resulting distribution will have mean and variance given by:

    m = Δ * (m1 / Δ1 + m2 / Δ2)
    Δ = 1 / (1 / Δ1 + 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting weight `c` is also computed and assigned to `g1`.
"""
function Base.prod!(g::gaussian_log_weight, g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    Δ = max(1 / (1 / Δ1 + 1 / Δ2), MIN_VAR)
    m = Δ * (m1 / Δ1 + m2 / Δ2)
    c = -(m1 - m2)^2 / (2 * (Δ1 + Δ2)) - log((sqrt(2 * pi * (Δ1 + Δ2))))
    g.mean = m
    g.var = max(Δ, MIN_VAR)
    g.log_weight = c + g1.log_weight + g2.log_weight
    # return gaussian_log_weight(m, Δ, c + g1.log_weight + g2.log_weight)
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
function Base.sum(g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    """
    Since we store the log weights (let's denote them by c_i) for know.
    We know that the weights are given by w_i = exp(c_i) / sum(exp(c_i)).
    Then, exp(c_i) cancels out to obtain the expression we use below
    """
    w1 = 1 / (1 + exp(g2.log_weight - g1.log_weight))
    # if isnan(w1)
    #     println(g1.log_weight, " ", g2.log_weight)
    #     println("Spotted w1 as NaN")
    # end
    w2 = 1 / (1 + exp(g1.log_weight - g2.log_weight))
    # if isnan(w2)
    #     println("Spotted w2 as NaN")
    # end
    m = m1 * w1 + m2 * w2
    Δ = w1 * (Δ1 + m1^2) + w2 * (Δ2 + m2^2) - m^2
    return gaussian_log_weight(m, Δ)
end

function nearest(g::gaussian_log_weight, y::Float64, h::Float64, e::Float64=1.0)
    m = g.mean
    rhs = -(m - y) * h
    b1 = floor(rhs)
    b2 = b1 + 1

    # The left and right Gaussian N_{L,i}, N_{R,i} in Liu eq 19
    gL = gaussian_log_weight(m - (b1 / h), g.var, g.log_weight)
    gR = gaussian_log_weight(m - (b2 / h), g.var, g.log_weight)

    # gL and gR are preprocessed as in Liu, eqs. 22-24 
    if ((y - e) < gL.mean < (y + e)) && !((y - e) < gR.mean < (y + e))
        gR = gL
    elseif ((y - e) < gR.mean < (y + e)) && !((y - e) < gL.mean < (y + e))
        gL = gR
    end
    return (gL, gR)
end


function nearest!(g1::gaussian_log_weight, g2::gaussian_log_weight, g::gaussian_log_weight, y::Float64, e::Float64=1.0)
    h = g.period
    m = g.mean
    rhs = (m - y) * h
    b1 = floor(rhs)
    b2 = b1 + 1

    left_mean = m - (b1 / h)
    right_mean = m - (b2 / h)

    g1.mean = left_mean
    g2.mean = right_mean
    g1.var = g.var
    g2.var = g.var
    g1.log_weight = g.log_weight
    g2.log_weight = g.log_weight

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


function m_nearest(g::gaussian_log_weight, y::Float64, h::Float64, M::Int64)
    h = g.period 
    m = g.mean
    rhs = (m - y) * h
    center = floor(rhs)
    offset = M ÷ 2
    gs = Vector{gaussian_log_weight}(undef, M)
    for k = 1:M
        b_k = center - (k - offset - 1)
        # println("b_$(k): ", b_k)
        gs[k] = gaussian_log_weight(m - (b_k / h), g.var)
    end
    return gs
end



"""
    m_nearest!(alloc::Vector{gaussian_log_weight}, g::gaussian_log_weight, y::Float64, M::Int64)
"""
function m_nearest!(alloc::Vector{gaussian_log_weight}, g::gaussian_log_weight, y::Float64, M::Int64)
    m = g.mean
    h = g.period
    rhs = (m - y) * h
    center = floor(rhs)
    offset = M ÷ 2
    @inbounds for k = 1:M
        b_k = center - (k - offset - 1)
        alloc[k].mean = m - (b_k / h)
        alloc[k].var = 1.0 * g.var
    end
end

"""
    sum!(g_out::gaussian_log_weigth, g1::gaussian_log_weigth, g2::gaussian_log_weigth)

Compute the sum of two Gaussian distributions `g1` and `g2` and store the result
in `g_out`. The resulting distribution will have mean and variance given by:

    m = m1 * w1 + m2 * w2
    Δ = w1 * (Δ1 + m1^2) + w2 * (Δ2 + m2^2) - m^2

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively, and `w1` and `w2` are weights that determine the contribution
of each distribution.
"""
function Base.sum!(g_out::gaussian_log_weight, g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    """
    Since we store the log weights (let's denote them by c_i) for know.
    We know that the weights are given by w_i = exp(c_i) / sum(exp(c_i)).
    Then, exp(c_i) cancels out to obtain the expression we use below
    """
    w1 = 1 / (1 + exp(g2.log_weight - g1.log_weight))
    w2 = 1 / (1 + exp(g1.log_weight - g2.log_weight))
    m = m1 * w1 + m2 * w2
    Δ = w1 * (Δ1 + m1^2) + w2 * (Δ2 + m2^2) - m^2

    g_out.mean = m
    g_out.var = max(Δ, MIN_VAR)
    g_out.log_weight = 0.0
    return nothing
end

"""
    moment_matching(gs::AbstractVector{gaussian_log_weigth})

A momment matching function written for gaussian_log_weigths input which weight represented in the log basis.
"""
function moment_matching(gs::AbstractVector{gaussian_log_weight})

    # raise exception if gs is empty
    if isempty(gs)
        error("gs is empty")
    end

    # Convert log_weights to normal weights and normalize in the same step. 
    ws = Vector{Float64}(undef, length(gs))
    for i in eachindex(ws)
        ws[i] = 1 / (sum(exp(g.log_weight - gs[i].log_weight) for g in gs))
    end

    # check that the weights sum to 1
    if !(sum(ws) ≈ 1.0)
        println("guassians are")
        println(gs)
        println("Weights do not sum to 1")
        println(ws)
        println(sum(ws))
        error("Weights do not sum to 1")
    end

    # Compute the mean
    m = sum([g.mean * w for (g, w) in zip(gs, ws)])

    # Compute the variance
    Δ = sum([w * (g.var + g.mean^2) for (g, w) in zip(gs, ws)]) - m^2
    if Δ < 0
        println(gs)
        error("Variance is negative")
    end

    return gaussian_log_weight(m, Δ)
end


"""
    moment_matching!(g::gaussian_log_weight, gs::AbstractVector{gaussian_log_weigth})

A momment matching function written for gaussian_log_weigths input which weight represented in the log basis.
"""
function moment_matching!(g::gaussian_log_weight, gs::AbstractVector{gaussian_log_weight})

    # raise exception if gs is empty
    if isempty(gs)
        error("gs is empty")
    end

    # Convert log_weights to normal weights and normalize in the same step. 
    ws = Vector{Float64}(undef, length(gs))
    for i in eachindex(ws)
        ws[i] = 1 / (sum(exp(g.log_weight - gs[i].log_weight) for g in gs))
    end

      # Compute weighted mean / variance without allocs
      m = 0.0
      Δ = 0.0
      for i in eachindex(gs)
          m += gs[i].mean * ws[i]
          Δ += ws[i] * (gs[i].var + gs[i].mean^2)
      end
      Δ -= m^2


    # # Compute the mean
    # m = sum([g.mean * w for (g, w) in zip(gs, ws)])

    # # Compute the variance
    # Δ = sum([w * (g.var + g.mean^2) for (g, w) in zip(gs, ws)]) - m^2
    # if Δ < 0
    #     println(gs)
    #     error("Variance is negative")
    # end

    g.mean = m
    g.var = max(Δ, MIN_VAR)
end


function moment_matching!(g::gaussian_log_weight, gs::AbstractVector{gaussian_log_weight}, ws::AbstractVector{Float64})
    @inbounds begin
        # Compute normalized weights in-place
        for i in eachindex(gs)
            # Compute denominator for normalization
            denom = 0.0
            for j in eachindex(gs)
                denom += exp(gs[j].log_weight - gs[i].log_weight)
            end
            ws[i] = 1 / denom
        end

        # Compute weighted mean
        m = 0.0
        for i in eachindex(gs)
            m += gs[i].mean * ws[i]
        end

        # Compute weighted variance
        Δ = - m^2
        for i in eachindex(gs)
            Δ += ws[i] * (gs[i].var + gs[i].mean^2)
        end

        if Δ < 0
            error("Variance is negative")
        end

        # Update g in-place
        g.mean = m
        g.var = ifelse(Δ < MIN_VAR, MIN_VAR, Δ)
    end
end




"""
    divide(g1::gaussian_log_weight, g2::gaussian_log_weight)

Compute the division of two Gaussian distributions `g1` and `g2`. The resulting
distribution will have mean and variance given by:

    m = Δ * (m1 / Δ1 - m2 / Δ2)
    Δ = 1 / (1 / Δ1 - 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting distribution has the correct weight `β`. 
The expression is obtained from http://davmre.github.io/blog/statistics/2015/03/27/gaussian_quotient

The function returns a new `gaussian_log_weight` distribution with weight `β`.

"""
function divide(g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    if Δ1 < MIN_VAR || Δ2 < MIN_VAR
        error("Variance is too small")
    end

    @assert Δ1 > 0 && Δ2 > 0
    if (Δ2 - Δ1) < 0
        println("Δ1: ", Δ1)
        println("Δ2: ", Δ2)
        println("g1: ", g1)
        println("g2: ", g2)
        error("Variance is negative")
        exit()
    end

    # Δ = 1 / (1 / Δ1 - 1 / Δ2)
    Δ = max(Δ1 * Δ2 / (Δ2 - Δ1), MIN_VAR)
    @assert Δ >= MIN_VAR
    # m = Δ * (m1 / Δ1 - m2 / Δ2)
    m = (Δ2 * m1 - Δ1 * m2) / (Δ2 - Δ1)
    log_β = (m1 - m2)^2 / (2.0 * (Δ2 - Δ1)) + 0.5 * log(2.0 * pi) + log(Δ2) - 0.5 * log(Δ2 - Δ1) + g1.log_weight - g2.log_weight
    return gaussian_log_weight(m, Δ, log_β)
end

"""
    divide!(g1::gaussian, g2::gaussian)

Compute the division of two Gaussian distributions `g1` and `g2` and store the result
in `g1`. The resulting distribution will have mean and variance given by:

    m = Δ * (m1 / Δ1 - m2 / Δ2)
    Δ = 1 / (1 / Δ1 - 1 / Δ2)

where `m1`, `m2`, `Δ1`, and `Δ2` are the mean and variance of `g1` and `g2`,
respectively. The resulting distribution has the correct weight `β`.
The expression is obtained from http://davmre.github.io/blog/statistics/2015/03/27/gaussian_quotient
"""
function divide!(g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var




    Δ = 1 / (1 / Δ1 - 1 / Δ2)
    m = Δ * (m1 / Δ1 - m2 / Δ2)
    log_β = (m1 - m2)^2 / (2 * (Δ2 - Δ1)) + 0.5 * log(2 * pi) + log(Δ2) - 0.5 * log(Δ2 - Δ1) + g1.log_weight - g2.log_weight

    g1.mean = m
    g1.var = Δ
    g1.log_weight = log_β
    return nothing
end


function divide!(g_out::gaussian_log_weight, g1::gaussian_log_weight, g2::gaussian_log_weight)
    m1 = g1.mean
    m2 = g2.mean
    Δ1 = g1.var
    Δ2 = g2.var

    if Δ1 < MIN_VAR || Δ2 < MIN_VAR
        error("Variance is too small")
    end

    @assert Δ1 > 0 && Δ2 > 0
    if (Δ2 - Δ1) < 0
        println("Δ1: ", Δ1)
        println("Δ2: ", Δ2)
        println("g1: ", g1)
        println("g2: ", g2)
        error("Variance is negative")
        exit()
    end

    Δ = 1 / (1 / Δ1 - 1 / Δ2)
    m = Δ * (m1 / Δ1 - m2 / Δ2)
    log_β = (m1 - m2)^2 / (2 * (Δ2 - Δ1)) + 0.5 * log(2 * pi) + log(Δ2) - 0.5 * log(Δ2 - Δ1) + g1.log_weight - g2.log_weight

    g_out.mean = m
    g_out.var = Δ
    g_out.log_weight = log_β
    return nothing
end


function Base.prod!(gs::Vector{gaussian_log_weight}, g::gaussian_log_weight)
    for k = 1:length(gs)
        prod!(gs[k], g)
    end
end


function Base.prod(gs::Vector{gaussian_log_weight}, g::gaussian_log_weight)
    terms = Vector{gaussian_log_weight}(undef, length(gs))
    for k = 1:length(gs)
        terms[k] = prod(gs[k], g)
    end
    return terms
end
Base.prod(g::gaussian_log_weight, gs::Vector{gaussian_log_weight}) = Base.prod(gs, g)

function Base.prod(gs1::Vector{gaussian_log_weight}, gs2::Vector{gaussian_log_weight})
    n1 = length(gs1)
    n2 = length(gs2)
    terms = Vector{gaussian_log_weight}(undef, n1 * n2)
    idx = 1
    for g1 in gs1
        for g2 in gs2
            terms[idx] = prod(g1, g2)
            idx += 1
        end
    end
    return terms
end





function Base.prod!(dest::AbstractVector{gaussian_log_weight}, gs::AbstractVector{gaussian_log_weight}, g::gaussian_log_weight)
    n1 = length(gs)
    for idx = 1:n1
        @views prod!(dest[idx], gs[idx], g)
    end
    return nothing
end

function Base.prod!(dest::AbstractVector{gaussian_log_weight}, gs1::AbstractVector{gaussian_log_weight}, gs2::AbstractVector{gaussian_log_weight})
    n1 = length(gs1)
    n2 = length(gs2)
    idx = 1
    for idx1 = 1:n1
        for idx2 = 1:n2
            @views prod!(dest[idx], gs1[idx1], gs2[idx2])
            idx += 1
        end
    end
    return nothing
end
