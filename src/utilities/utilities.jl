using Random


"""
    sample_error(σ::Float64, n::Int)

Sample a Gaussian error vector with standard deviation `σ` and length `n`.
"""
function sample_error(σ::Float64, n::Int)
    return σ * randn(n)
end



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