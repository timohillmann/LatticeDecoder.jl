using LinearAlgebra
using Combinatorics
using LLLplus
using Nemo

struct LocalSearch
    w::Int64
    G::AbstractMatrix{Float64}
    lll_reduction::Bool
    order::Vector{Int64}
    candidates::Vector{Vector{Int64}}
    sphere_decoding::Bool
    full_basis::Bool
end

function LocalSearch(w::Int64, G::AbstractMatrix{Float64}, order::Vector{Int64}, lll_reduction::Bool, sphere_decoding::Bool, full_basis::Bool)

    candidates = full_basis ? Vector{Vector{Int64}}() : generate_candidates(w, order)
    return LocalSearch(w, G, lll_reduction, order, candidates, sphere_decoding, full_basis)
end 


function generate_value_sets(rs::Vector{Int64}, i::Int64)
        return Iterators.product(fill(rs, (1, i))...)
end


function generate_candidates(n::Int64, order::Vector{Int64})
    vecs = Vector{Vector{Int64}}()
    _vec = zeros(Int64, n)
    for i in eachindex(order)
        r = order[i]
        rs = [s * _r for _r in 1:r if r != 0 for s in [-1, 1]]
        # printstyled("rs: $rs\n", color=:red)
        for r in generate_value_sets(rs, i)  # multiset_combinations(rs, i)
            # i could change the range 1:n to contain only a subset of the indices based on soft information
            # printstyled("r: $r\n", color=:blue)
            for idx in multiset_combinations(1:n, i)
                # printstyled("idx: $idx\n", color=:green)
                _vec .= 0
                _vec[idx] .= r
                push!(vecs, copy(_vec))
            end
        end
    end

    return vecs
end

function babai_nearest_plane(B::AbstractMatrix{Float64}, r::Vector{Float64})
    # println(size(B))
    # println(size(r))
    # Use a numerically stable least-squares solve instead of normal equations.
    return round.(B \ r)
end


function local_search!(y::Vector{Float64}, λ::Vector{Float64}, dec::Vector{Int64},lsd::LocalSearch)
    if lsd.full_basis
        S = 1:size(lsd.G, 2)
    else
        S = select_basis(λ, lsd.w)
    end
    B = lsd.G[:, S]  # select unreliable columns
    r = y - lsd.G * dec

    if lsd.sphere_decoding
        if lsd.lll_reduction
            B, T, _Q, _R = lll_reduce(B)
            u = sphere_decode_small(B, r)
            dec[S] .+= T * u
        else
            u = sphere_decode_small(B, r)
            dec[S] .+= u
        end

    else
        if lsd.lll_reduction
            B, T, _Q, _R = lll_reduce(B)
            u_0 = babai_nearest_plane(B, r)

            u = T * (u_0 + local_cvp(B, r - B * u_0, lsd))
        else
            u_0 = babai_nearest_plane(B, r)
            u = u_0 + local_cvp(B, r - B * u_0, lsd)
        end
        dec[S] .+= u
    end

    return nothing
end

function local_cvp(A::AbstractMatrix{Float64}, r::Vector{Float64}, lsd::LocalSearch)
    best_u = zeros(Int64, size(A, 2))
    best_dist = sum(abs2, r)
    for candidate in lsd.candidates
        dist = sum(abs2, r - A * (candidate))
        if dist < best_dist
            best_dist = dist
            best_u .= candidate
            # println("Updated best_dist: $(best_dist)")
        end
    end
    return best_u
end 


function select_basis(λ::Vector{Float64}, n_faults::Int64)
    sort_idx = sortperm(λ, rev=true)
    n = min(n_faults, length(sort_idx))
    return sort_idx[1:n]
end

function solution_weight(b::Vector{Float64})
    return norm(b)
end


function lll_reduce(B::Matrix{Float64}; δ::Union{Nothing,Float64}=nothing)
    if δ === nothing
        B, T, _ = LLLplus.lll(B)
    else
        B, T, _ = LLLplus.lll(B, δ)
    end
    Q, Rfactor = qr(B)
    R = Matrix(Rfactor)
    # n, t = size(B)
    # scale = 10^6
    # # Construct a Nemo integer matrix from B
    # ZZ = Nemo.ZZ
    # M = Nemo.Matrix(ZZ, n, t)
    # for i in 1:n, j in 1:t
    #     M[i, j] = ZZ(round(Int, scale * B[i, j]))
    # end
    # # Run LLL
    # M_lll, U = Nemo.lll_with_transform(M)
    # # Convert reduced basis and unimodular matrix back to Float64
    # B_red = Float64.(Array(M_lll)) / scale
    # U_red = Int.(Array(U))
    return B, T, Q, R
end

using LinearAlgebra

"""
    sphere_decode_small(B::AbstractMatrix{<:Real}, r::AbstractVector{<:Real};
                       radius2::Union{Nothing,Real}=nothing,
                       max_nodes::Integer=10^7,
                       use_babai::Bool=true)

Solve min_{u ∈ Z^t} || r - B*u ||^2 using a depth-first sphere decoder optimized for small t.

Arguments
---------
- B : n×t real matrix (basis)
- r : n-vector (target)
- radius2 : optional initial radius squared (if `nothing`, we use Babai distance if available; otherwise +Inf)
- max_nodes : maximum number of visited nodes (safeguard)
- use_babai : whether to compute Babai init and use it as starting radius and initial best

Returns
-------
(best_u::Vector{Int}, best_dist2::Float64, nodes_visited::Int)
- best_u: integer vector achieving best found distance
- best_dist2: squared distance ||r - B*best_u||^2
- nodes_visited: number of tree nodes visited
"""
function sphere_decode_small(B::AbstractMatrix{<:Real}, r::AbstractVector{<:Real};
                             radius2::Union{Nothing,Real}=nothing,
                             max_nodes::Integer=10^7,
                             use_babai::Bool=true)

    # println("Running sphere_decode_small")
    # sizes
    n, t = size(B)

    # QR: B = Q * R  (thin QR)
    F = eltype(B)
    Q, Rfactor = qr(B)          # Q is n×t, Rfactor is t×t upper triangular
    R = Matrix(Rfactor)         # make dense matrix type-stable
    y = Matrix(Q)' * r          # dimension t vector: projection Q' * r
    y = vec(y)

    # precompute diagonal entries and their squares for pruning
    Rdiag = [R[i,i] for i in 1:t]

    # Babai initialization (back-substitution rounding in R)
    function babai_on_R(R, y)
        u = zeros(Int, t)
        for i in t:-1:1
            s = zero(Float64)
            if i < t
                @inbounds for j = i+1:t
                    s += R[i,j] * u[j]
                end
            end
            u[i] = round(Int, (y[i] - s) / R[i,i])
        end
        return u
    end

    best_u = zeros(Int, t)
    if use_babai
        u_babai = babai_on_R(R, y)
        # compute distance squared for Babai
        res = y - R * Float64.(u_babai)
        d_babai = sum(abs2, res)
        # seed
        best_u .= u_babai
        best_dist2 = radius2 === nothing ? d_babai : min(d_babai, radius2)
    else
        best_dist2 = radius2 === nothing ? Inf : radius2
    end

    if radius2 !== nothing
        best_dist2 = min(best_dist2, float(radius2))
    end

    # recursion state (mutable arrays to avoid allocations)
    u = zeros(Int, t)                # current partial integer vector
    partial = zeros(Float64, t+1)    # partial squared metrics; partial[k+1] = sum_{i=k+1..t} ...
    # We'll interpret level index k in {t, t-1, ..., 1}.
    nodes = Ref{Int64}(0)                   # visited node counter (mutable)

    # Helper: enumerate integer candidates in order round(center) ± 0,1,2,...
    # yields generator of integers in that order; implemented as closure returning successive values
    function candidate_sequence(center::Float64)
        base = round(Int, center)
        # return an iterator that yields base, base+1, base-1, base+2, base-2, ...
        return Iterators.flatten((Iterators.map(x -> base + x, (0,1,-1,2,-2,3,-3,4,-4,5,-5)))) # long enough for small t
    end

    # To avoid infinite sequences, we instead implement manual loop in recursion that tries offsets 0,±1,... until metric exceeds bound.
    # Recursive DFS function:
    function dfs(level::Int)
        nodes[] += 1
        if nodes[] >= max_nodes
            return
        end
        if level == 0
            # reached leaf; update best if partial metric < best
            nodes[] += 1
            if partial[1] < best_dist2
                best_dist2 = partial[1]
                best_u .= u
            end
            return
        end

        # compute the center (continuous coordinate) for this level given already chosen u[level+1..t]
        # center = (y[level] - sum_{j=level+1..t} R[level,j] * u[j]) / R[level,level]
        s = 0.0
        @inbounds for j = level+1:t
            s += R[level, j] * u[j]
        end
        center = (y[level] - s) / Rdiag[level]

        # we will try integers m = round(center) + offset in increasing |offset|
        m0 = round(Int, center)

        # compute squared contribution from difference at this level:
        # contrib(m) = (Rdiag[level] * (m - center))^2
        # remaining metric after choosing m equals contrib(m) + partial[level+1]
        # if contrib(m) + partial[level+1] >= best_dist2 then prune this branch.

        # iterate offsets 0, +1, -1, +2, -2, ... until pruning or reasonable bound
        offset = 0
        sign = 1
        while true
            m = m0 + ((offset==0) ? 0 : sign*offset)
            diff = (m - center)
            contrib = (Rdiag[level] * diff) ^ 2
            metric_here = contrib + partial[level+1]
            if metric_here < best_dist2 - 1e-15   # allow tiny eps
                # accept candidate and go deeper
                u[level] = m
                partial[level] = metric_here
                dfs(level - 1)
                # restore not necessary: u[level] overwritten next loop
            end

            # prepare next offset ordering
            if offset == 0
                offset = 1
                sign = 1
            else
                if sign == 1
                    sign = -1
                else
                    sign = 1
                    offset += 1
                end
            end

            # cheap pruning: If even the best possible candidate (with diff=0) at next offset produces contrib >= best_dist2 - partial[level+1], we can break.
            # That is: minimal possible contrib is 0 (when m==round(center)), but as offsets grow, contrib grows; break when even offset==0 was pruned AND offset>some limit
            # To be safe, stop when: (Rdiag[level] * (offset - 0.5))^2 + partial[level+1] >= best_dist2
            # Using (offset-0.5) as approximate lower bound for |m-center| at that offset
            approx_lower = (Rdiag[level] * max(0.0, offset - 0.5))^2 + partial[level+1]
            if approx_lower >= best_dist2 - 1e-15
                break
            end

            # safety cap (avoid infinite loop): if offset grows too large relative to best_dist2, break
            if offset > 1000
                break
            end

            if nodes[] >= max_nodes
                break
            end
        end
    end

    # start recursion at top level = t
    # partial[t+1] = 0, partial[1] will contain full metric at leaf
    partial[t+1] = 0.0

    # initial check: if best_dist2 is Inf and no Babai init, set a conservative huge radius to allow search
    if isinf(best_dist2)
        best_dist2 = typemax(Float64)
    end

    dfs(t)

    return copy(best_u)
end


# H = [[1, 0, 0, 0, 0] [1 / 2, 1 / 2, 0, 0, 0] [0, 1 / 2, 1 / 2, 0, 0] [0, 0, 1 / 2, 1 / 2, 0] [0, 0, 0, 1 / 2, 1 / 2]]
# G = inv(H)
# n = 5
# y = 0.5 * randn(n)
# b = [0, 0, 0, 0, 0]
# λ = abs.(y - G * b)

# """
#     hard_decision(bp_result::Vector{Float64}, H::AbstractArray)

# Compute the hard decision using the belief propagation result.
# """
# function hard_decision(bp_result::Vector{Float64}, H::AbstractArray)
#     return Int64.(round.(H * bp_result))
# end

# dec = hard_decision(y, H)

# w = 5
# order = [2, 1, 1]

# lsd = LocalSearch(w, G, order, true);


# println(dec)
# local_search!(y, λ, dec, lsd)
# println(dec)

# # """

# OSD-style postprocessing (small-CVP variant)

# Input: G, H, y, BP candidate \tilde x, parameters t, L (optional)

# 1. a0 := round(H * \tilde x)                         # nearest integers
# 2. Compute reliabilities r_k = distance of (H \tilde x)_k to nearest integer
# 3. Choose S = indices of top-t largest r_k (or top-t posterior variances)
# 4. x0 := G * a0
# 5. B := G[:, S]                # n x t matrix (columns of G for S)
# 6. r := y - x0
# 7. Compute initial u0:
#      - Solve real least-squares u_rls = argmin_u || r - B u ||_2  (u_rls = (B^T B)^{-1} B^T r)
#      - u_init := round(u_rls)  (or Babai nearest-plane on basis B)
# 8. If using bounded search:
#      - restrict u_i ∈ {u_init_i - L, ..., u_init_i + L} for each i
#      - enumerate candidates (cartesian product) or guided search
#    Else:
#      - run sphere decoder (QR of B) centered at u_init with radius R = ||r - B u_init||^2
# 9. For each candidate u, compute a = a0 + E_S u, x = G a and distance d = ||y - x||^2
# 10. Return candidate x with minimum d (and its a)
# """
