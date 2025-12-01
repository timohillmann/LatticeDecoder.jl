# using Oscar, Combinatorics, LinearAlgebra, Random, Nemo

# """
#     construct_generator_opt(Gint::Matrix{Int}, p::Int; max_samples::Int=1000)

# Construct a generator matrix following the "min-det subset" procedure over GF(p),
# with a fallback heuristic for large matrices.

# Arguments:
# - Gint: k x n integer matrix (entries modulo p)
# - p: prime number (field)
# - max_samples: max number of random subsets to sample in heuristic mode

# Returns:
# - Gexp: the final integer generator matrix with extra rows
# - best_cols: selected column indices
# - best_rows: selected row indices
# """
# function construct_generator_opt(Gint::Matrix{Int}, p::Int; max_samples::Int=1000)
#     k, n = size(Gint)
#     F, _ = residue_ring(ZZ, p)
#     G = matrix(F, Gint)
#     r = rank(G)

#     # Step 2: select r columns minimizing det(S^T S)
#     best_det = Inf
#     best_cols = nothing
#     all_combs = collect(Combinatorics.combinations(1:n, r))
#     sampled_combs = length(all_combs) > max_samples ? 
#         randperm(length(all_combs))[1:max_samples] : 1:length(all_combs)

#     for idx in sampled_combs
#         println(idx)
#         cols = collect(all_combs[idx])
#         S = deepcopy(G[:, cols])
#         S_int = to_int(S)
#         det_val = det(S_int' * S_int)
#         if det_val < best_det
#             best_det = det_val
#             best_cols = cols
#         end
#     end
#     S = G[:, best_cols]

#     # Step 3: remove k-r rows to keep full rank
#     best_rows = nothing
#     all_row_combs = collect(Combinatorics.combinations(1:k, r))
#     sampled_rows = length(all_row_combs) > max_samples ?
#         randperm(length(all_row_combs))[1:max_samples] : 1:length(all_row_combs)

#     best_det_rows = Inf
#     for idx in sampled_rows
#         rows = collect(all_row_combs[idx])
#         Gprime = G[rows, :]
#         if rank(Gprime) == r
#             Sprime_int = to_int(deepcopy(Gprime[:, best_cols]))
#             det_val = det(Sprime_int' * Sprime_int)
#             if det_val < best_det_rows
#                 best_det_rows = det_val
#                 best_rows = rows
#             end
#         end
#     end
#     Gprime = G[best_rows, :]

#     # Step 4: expand G' with n-r rows
#     unselected = setdiff(1:n, best_cols)
#     extra = zeros(Int, length(unselected), n)
#     for (i, c) in enumerate(unselected)
#         extra[i, c] = p
#     end

#     # Step 5: lift G' to integers and append extra rows
#     Gexp =to_int(Gprime)
#     Gexp = vcat(Gexp, extra)

#     return Gexp, best_cols, best_rows
# end

using LinearAlgebra
using Combinatorics

# ========== Basic utilities ==========

"""
    modp_rank(M::AbstractMatrix{<:Integer}, p::Int)

Compute the rank of a matrix M over GF(p) using Gaussian elimination mod p.
"""
function modp_rank(M::AbstractMatrix{<:Integer}, p::Int)
    A = copy(M) .% p
    m, n = size(A)
    r = 0
    col = 1
    for row in 1:m
        while col <= n && all(A[row:end, col] .== 0)
            col += 1
        end
        col > n && break
        # find pivot
        pivot_row = findfirst(!iszero, A[row:end, col])
        pivot_row !== nothing || continue
        pivot_row += row - 1
        # swap if needed
        if pivot_row != row
            A[row, :], A[pivot_row, :] = A[pivot_row, :], A[row, :]
        end
        # normalize pivot
        pivot_inv = invmod(A[row, col], p)
        A[row, :] = (A[row, :] .* pivot_inv) .% p
        # eliminate below
        for r2 in (row+1):m
            if A[r2, col] != 0
                A[r2, :] = (A[r2, :] .- A[r2, col] .* A[row, :]) .% p
            end
        end
        r += 1
        col += 1
    end
    return r
end

"""
    modp_is_full_rank(M::AbstractMatrix{<:Integer}, p::Int)

Return true if matrix M has full row rank over GF(p).
"""
modp_is_full_rank(M::AbstractMatrix{<:Integer}, p::Int) = modp_rank(M, p) == size(M, 1)

# ========== Column selection ==========

function select_columns_modp(G::Matrix{Int}, r::Int, p::Int)
    k, n = size(G)
    best_cols = nothing
    best_score = typemax(Int)
    for cols in combinations(1:n, r)
        S = G[:, collect(cols)] .% p
        if modp_rank(S, p) == r
            score = sum(S)  # heuristic: minimize total sum
            if score < best_score
                best_score = score
                best_cols = collect(cols)
            end
        end
    end
    best_cols === nothing && error("Cannot find full-rank columns mod $p")
    return best_cols
end

# ========== Row selection ==========

function select_rows_modp(G::Matrix{Int}, kept_cols::Vector{Int}, r::Int, p::Int)
    k, _ = size(G)
    best_rows = nothing
    best_score = typemax(Int)
    for rows in combinations(1:k, r)
        Gsub = G[collect(rows), :] .% p
        if modp_rank(Gsub, p) == r
            score = sum(Gsub[:, kept_cols])
            if score < best_score
                best_score = score
                best_rows = collect(rows)
            end
        end
    end
    best_rows === nothing && error("Cannot find suitable rows mod $p")
    return best_rows
end

# ========== Main algorithm ==========

"""
    expand_matrix_modp(G::Matrix{Int}, p::Int)

Generalization to any prime field p:
1. Compute rank r over GF(p)
2. Select r columns that yield full-rank submatrix and minimal weight
3. Remove k - r rows while keeping full rank
4. Add (n - r) rows with entry p in unselected columns
"""
function expand_matrix_modp(G::Matrix{Int}, p::Int)
    k, n = size(G)
    r = modp_rank(G, p)
    @assert r > 0 "Matrix has rank 0 mod $p"

    cols = select_columns_modp(G, r, p)
    println(cols)
    rows = select_rows_modp(G, cols, r, p)

    Gprime = G[rows, :]  # keep selected rows

    unselected = setdiff(1:n, cols)
    extra = zeros(Int, n - r, n)
    for (i, c) in enumerate(unselected)
        extra[i, c] = p
    end

    Gexp = vcat(Gprime, extra)

    info = Dict(
        :rank => r,
        :selected_cols => cols,
        :selected_rows => rows,
        :unselected_cols => unselected
    )

    return Gexp, info
end


# Example usage
G = [
    1 1 1 0 1 0 0;
    0 1 1 1 0 1 0;
    0 0 1 1 1 0 1
]

G = [ 
    1 0 1 1 0 0 0;
    0 1 0 1 1 0 0;
    0 0 1 0 1 1 0;
    0 0 0 1 0 1 1
]

p = 2

Gexp, info = expand_matrix_modp(G, p)
println("Original G:\n")
display(G)
println("\nExpanded G':\n")
display(Gexp)
println("\nInfo:\n", info)

det(Gexp)