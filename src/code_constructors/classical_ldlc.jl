using Random
using Logging
using LinearAlgebra

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
        if counter == 10
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

