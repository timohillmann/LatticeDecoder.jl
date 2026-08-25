using Random


"""
    sample_error([rng], σ, n)

Sample a Gaussian error vector with standard deviation `σ` and length `n`.
Pass an explicit random-number generator for reproducible simulations.
"""
function sample_error(rng::AbstractRNG, σ::Real, n::Integer)
    isfinite(σ) && σ >= 0 || throw(ArgumentError("noise standard deviation must be finite and nonnegative"))
    n >= 0 || throw(ArgumentError("sample length must be nonnegative"))
    return Float64(σ) .* randn(rng, Int(n))
end

sample_error(σ::Real, n::Integer) = sample_error(Random.default_rng(), σ, n)



"""
    hard_decision(bp_result::Vector{Float64}, H::AbstractArray)

Compute the hard decision using the belief propagation result.
"""
function hard_decision(bp_result::Vector{Float64}, H::AbstractArray)
    return Int64.(round.(H * bp_result))
end


"""
    count_symbol_errors(x::AbstractArray)

Count the number of bit errors in the decoded codeword. The codeword is assumed to be the all-zero vector.
"""
function count_symbol_errors(x::AbstractArray)
    return sum(x .!= 0)
end


"""
    count_symbol_errors(x::AbstractArray, y::AbstractArray)

Count the number of bit errors between two codewords x and y.
"""
function count_symbol_errors(x::AbstractArray, y::AbstractArray)
    return sum(x .!= y)

end


"""
    lattice_capacity_var()

Return the variance of the Gaussian noise that achieves the capacity of the lattice code.
"""
function lattice_capacity_var()
    return 1 / (2 * pi * ℯ)
end

"""
    lattice_capacity_std()

Return the standard deviation of the Gaussian noise that achieves the capacity of the lattice code.
"""
function lattice_capacity_std()
    return 1 / sqrt(2 * pi * ℯ)
end


"""
    signal_to_noise_ratio(σ::Float64)

Return the signal-to-noise ratio (SNR) for a given standard deviation `σ`.
"""
function signal_to_noise_ratio(σ::Float64)
    return 1 / (2 * pi * exp(1) * σ^2)
end

"""
    snr(σ::Float64)

Return the signal-to-noise ratio (SNR) for a given standard deviation `σ`.
"""
snr(σ::Float64) = signal_to_noise_ratio(σ)

"""
    snr_db(σ::Float64)

Return the signal-to-noise ratio (SNR) in decibels for a given standard deviation `σ`.
"""
function snr_db(σ::Float64)
    return 10 * log10(snr(σ))
end


"""
    symplectic_form(n::Int)

Return the symplectic form for `n` modes in the `qqpp` basis.
"""
function symplectic_form(n::Int)
    return kron(Float64[0 1; -1 0], Matrix{Float64}(I, n, n))
end


"""
    symbol_error_rate_rounding(n::Int64, σ::Float64)

Return the symbol error rate for a given number of symbols `n` and standard deviation `σ`.
"""
function symbol_error_rate_rounding(n::Int64, σ::Float64)
    return 1 - (1 - 2 * normcdf(-1 / (2 * σ)))^n
end


"""
    symbol_error_rate_hard_decision(n::Int64, σ::Vector{Float64})

Return the symbol error rate for a given number of symbols `n` and standard deviation `σ`.
"""
function symbol_error_rate_hard_decision(n::Int64, σ::Vector{Float64})
    return 1 - (1 - 2 * normcdf(-1 ./ (2 * σ)))^n
end


"""
    agresti_coull_confidence_interval(p, n, z=1.96)

Compute the Agresti-Coull confidence interval for a binomial distribution.
The default value of z is for a 95% confidence interval.
To get the 99% confidence interval, use z=2.576.
To get the 99.9% confidence interval, use z=3.291.

"""
function agresti_coull_confidence_interval(p, n, z=1.96)
    p_ = (n * p + z^2 / 2) / (n + z^2)
    n_ = n + z^2
    return z * sqrt(p_ * (1 - p_) / n_)
end

"""
    is_logical_error(logical_check, residual; atol=1e-5)

Return `true` when `residual` has a nonintegral logical coordinate. The rows
of `logical_check` span the dual checks used to classify residual shifts.
"""
function is_logical_error(
    logical_check::AbstractMatrix,
    residual::AbstractVector;
    atol::Real=1e-5,
)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    size(logical_check, 1) == length(residual) ||
        throw(DimensionMismatch("logical_check must have one row per residual coordinate"))
    coordinates = logical_check' * residual
    return any(abs(value - round(value)) >= atol for value in coordinates)
end



random_bitstring(rng::AbstractRNG, n::Integer) = rand(rng, 0:1, n)
random_bitstring(n::Integer) = random_bitstring(Random.default_rng(), n)

function random_bitstring!(rng::AbstractRNG, b::Vector{Int64}, n::Integer=length(b))
    0 <= n <= length(b) || throw(ArgumentError("n must be between zero and length(b)"))
    @inbounds for i = 1:n
        b[i] = rand(rng, 0:1)
    end
    return b
end

random_bitstring!(b::Vector{Int64}, n::Integer=length(b)) =
    random_bitstring!(Random.default_rng(), b, n)
