using Random
using Logging
using LinearAlgebra
using NPZ

"""
    init_p_mat(d::Int, n::Int)

Initialize a random permutation matrix P of size d x n.
"""
function init_p_mat(d::Int, n::Int)
    P = zeros(Int, d, n)
    for kk in 1:d
        P[kk, :] .= randperm(n)
    end
    return P
end


"""
    build_H_mat(d::Int, n::Int, h::Vector{Float64}, P::AbstractMatrix)

Builds the H matrix from the permutation matrix P and the vector of weights h.
"""
function build_H_mat(d::Int, n::Int, h::Vector{Float64}, P::AbstractMatrix)
    H = zeros(n, n)
    for i = 1:n
        for j = 1:d
            H[P[j, i], i] = h[j] * rand([-1, 1])
        end
    end
    return H
end

function first_two_common_elements!(s::Vector{Int64}, a::AbstractArray, b::AbstractArray)
    i = 1
    @inbounds for a_i ∈ a
        for b_i ∈ b
            if a_i == b_i
                s[i] = a_i
                i += 1
                if i == 2
                    return true
                end
            end
        end
    end
    return false
end

"""
    loop_removal!(P::AbstractMatrix, d::Int, n::Int)

    Remove loops of size 2 and 4 from the permutation matrix P. This function uses a random
    permutation to remove loops and thus potentially does not succeed in removing all loops.
    It modifies the input matrix P in place and returns a boolean indicating whether the
    loop removal was successful.
"""
function loop_removal!(P::AbstractMatrix, d::Int, n::Int)
    c = 1
    loopless_columns = 0
    max_iters = 100 * n
    iters = 0
    while loopless_columns < n
        changed_perm = 0
        # check P for 2-loops
        if any(P[i, c] == P[j, c] for i in 1:d for j in (i+1):d)
            for i in 1:d
                for j in (i+1):d
                    if P[i, c] == P[j, c]
                        changed_perm = i
                        break
                    end
                end
            end
        else
            for c0 in 1:n
                if c0 != c
                    common_elems = intersect(P[:, c], P[:, c0])
                    if length(common_elems) >= 2
                        # A 4-loop was found at column c
                        for i in 1:d
                            if P[i, c] == common_elems[1]
                                changed_perm = i
                                break
                            end
                        end
                    end
                end
            end
        end
        if changed_perm != 0 # a permutation should be modified to remove loop
            i = rand(1:n) # choose a random integer 1 ≤ i ≤ n
            P[changed_perm, [c, i]] = P[changed_perm, [i, c]] # swap locations c and i in permutation
            loopless_columns = 0 # reset loopless columns counter
        else
            # no loop war found at column c
            loopless_columns += 1
        end
        c += 1
        if c > n
            c = 1
        end
        iters += 1
        if iters == max_iters
            return false
            break
        end
    end
    return true
end






function updated_loop_removal!(P::AbstractMatrix, d::Int, n::Int)
    c = 1
    loopless_columns = 0
    max_iters = n
    iters = 0
    common_elems = Vector{Int64}(undef, 2)
    while loopless_columns < n
        changed_perm = 0
        # check P for 2-loops
        found_2_loop = false
        @inbounds for i in 1:d
            for j in (i+1):d
                if @views P[i, c] == P[j, c]
                    changed_perm = i
                    found_2_loop = true
                    break
                end
            end
        end
        @inbounds if !found_2_loop
            for c0 in 1:n
                if c0 != c
                    # common_elems = intersect(P[:, c], P[:, c0])
                    flag = first_two_common_elements!(common_elems, P[:, c], P[:, c0])
                    if flag
                        # A 4-loop was found at column c
                        for i in 1:d
                            if P[i, c] == common_elems[1]
                                changed_perm = i
                                break
                            end
                        end
                    end
                end
            end
        end
        @inbounds if changed_perm != 0 # a permutation should be modified to remove loop
            i = rand(1:n) # choose a random integer 1 ≤ i ≤ n
            @views P[changed_perm, [c, i]] = P[changed_perm, [i, c]] # swap locations c and i in permutation
            loopless_columns = 0 # reset loopless columns counter
        else
            # no loop war found at column c
            loopless_columns += 1
        end
        c += 1
        if c > n
            c = 1
        end
        iters += 1
        if iters == max_iters
            return false
            break
        end
    end
    return true
end


"""
    classical_ldlc(d::Int, n::Int, weights::Vector{Float64})

    Construct a classical LDLC code with d rows and n columns. The weights vector determines the
    edge weights of the H matrix.

    If `normalize` is set to true, the determinant of the H is such that abs(det(H)) = 1.
"""
function classical_ldlc(d::Int, n::Int, weights::Vector{Float64}, normalize::Bool=false)

    @assert length(weights) == d "The length of the weight vector must be equal to the desired weight of rows in the H matrix."

    converged = false
    counter = 0
    while converged == false
        global P = init_p_mat(d, n)

        converged = loop_removal!(P, d, n)
        counter += 1
        if counter == 1000
            break
        end
    end

    # If loop removal was not successful, throw a warning
    if !converged
        @warn "Warning: loop removal was not successful."
    end

    H = build_H_mat(d, n, weights, P)
    if normalize
        H = H / abs(det(H))^(1 / n)
    end

    return H
end


"""
    classical_ldlc(d::Int, n::Int)

    Construct a classical LDLC code with degree d over n bits. The edge weights of the H matrix are
    set to 1/sqrt(d) apart from a single entry which is set to 1.
"""
function classical_ldlc(d::Int, n::Int, normalize::Bool=false)
    h = [1.0]
    for _ in 1:d-1
        push!(h, 1 / sqrt(d))
    end
    return classical_ldlc(d, n, h, normalize)
end


"""
    generator_matrix(H::AbstractMatrix)

    Compute the generator matrix of the LDLC code from the parity-check matrix H.
"""
function generator_matrix(H::AbstractMatrix)
    return inv(H)
end


"""
    encode(x::AbstractArray, G::AbstractMatrix)

    Encode the input vector x using the generator matrix G.
"""
function encode(x::AbstractArray, G::AbstractMatrix)
    return G * x
end


"""
    encode!(x::AbstractArray, G::AbstractMatrix)

    Encode the input vector x using the generator matrix G. The result is stored in the input vector x.
"""
function encode!(x::AbstractArray, G::AbstractMatrix)
    x .= G * x
end


"""
    decode(x::AbstractArray, H::AbstractMatrix)

    Decode the input vector x using the parity-check matrix H.
"""
function decode(x::AbstractArray, H::AbstractMatrix)
    return Int64.(round.(H * x))
end


"""
    decode!(x::AbstractArray, H::AbstractMatrix)

    Decode the input vector x using the parity-check matrix H. The result is stored in the input vector x.
"""
function decode!(x::AbstractArray, H::AbstractMatrix)
    x .= Int64.(round.(H * x))
end


struct LDLCode
    H::AbstractMatrix{Float64}
end


"""
    generator_matrix(H::AbstractMatrix)

    Compute the generator matrix of the LDLC code from the parity-check matrix H.
"""
function generator_matrix(H::AbstractMatrix)
    return inv(H)
end


"""
    encode(x::AbstractArray, G::AbstractMatrix)

    Encode the input vector x using the generator matrix G.
"""
function encode(x::AbstractArray, G::AbstractMatrix)
    return G * x
end


"""
    encode!(x::AbstractArray, G::AbstractMatrix)

    Encode the input vector x using the generator matrix G. The result is stored in the input vector x.
"""
function encode!(x::AbstractArray, G::AbstractMatrix)
    x .= G * x
end


"""
    decode(x::AbstractArray, H::AbstractMatrix)

    Decode the input vector x using the parity-check matrix H.
"""
function decode(x::AbstractArray, H::AbstractMatrix)
    return Int64.(round.(H * x))
end


"""
    decode!(x::AbstractArray, H::AbstractMatrix)

    Decode the input vector x using the parity-check matrix H. The result is stored in the input vector x.
"""
function decode!(x::AbstractArray, H::AbstractMatrix)
    x .= Int64.(round.(H * x))
end


struct LDLCode
    H::AbstractMatrix{Float64}
end


function load_ldlc(d, n)
    code = npzread("/Users/timo/Documents/GitHub/LatticeDecoder.jl/data/generator_matrices/ldlc/d_$(d)_n_$(n).npz")
    return code
end

