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
    count_bit_errors(x::AbstractArray)

Count the number of bit errors in the decoded codeword. The codeword is assumed to be the all-zero vector.
"""
function count_bit_errors(x::AbstractArray)
    return sum(x .!= 0)
end


"""
    count_bit_errors(x::AbstractArray, y::AbstractArray)

Count the number of bit errors between two codewords x and y.
"""
function count_bit_errors(x::AbstractArray, y::AbstractArray)
    return sum(x .!= y)

end


function lattice_capacity_var()
    return 1 / (2 * pi * ℯ)
end

function lattice_capacity_std()
    return 1 / sqrt(2 * pi * ℯ)
end