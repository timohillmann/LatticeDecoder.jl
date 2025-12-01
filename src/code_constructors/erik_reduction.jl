using LinearAlgebra
using Combinatorics

# ========== Basic GF(2) utilities ==========

"""
    gf2_rank(M::AbstractMatrix{<:Integer})

Compute the rank of binary matrix M over GF(2).
Uses Gaussian elimination mod 2.
"""
function gf2_rank(M::AbstractMatrix{<:Integer})
    A = copy(M) .% 2
    m, n = size(A)
    r = 0
    col = 1
    for row in 1:m
        while col ≤ n && all(A[row:end, col] .== 0)
            col += 1
        end
        col > n && break
        # find pivot
        pivot_row = findfirst(!iszero, A[row:end, col])
        if pivot_row !== nothing
            pivot_row += row - 1
            # swap if needed
            if pivot_row != row
                A[row, :], A[pivot_row, :] = A[pivot_row, :], A[row, :]
            end
            # eliminate below
            for r2 in (row+1):m
                if A[r2, col] == 1
                    A[r2, :] = (A[r2, :] .⊻ A[row, :]) .% 2
                end
            end
            r += 1
        end
        col += 1
    end
    return r
end

"""
    gf2_is_full_rank(M::AbstractMatrix{<:Integer})

Return true if matrix M has full row rank over GF(2).
"""
gf2_is_full_rank(M::AbstractMatrix{<:Integer}) = gf2_rank(M) == size(M, 1)

# ========== Step 2: Column selection ==========

"""
    select_columns_gf2(G::Matrix{Int}, r::Int)

Select r columns that make G[:, cols] have full rank over GF(2),
preferring subsets with minimal total number of 1s.
"""
function select_columns_gf2(G::Matrix{Int}, r::Int)
    k, n = size(G)
    best_cols = nothing
    best_score = typemax(Int)
    for cols in combinations(1:n, r)
        S = G[:, collect(cols)]
        if gf2_rank(S) == r
            score = sum(S)  # minimize number of 1s
            if score < best_score
                best_score = score
                best_cols = collect(cols)
            end
        end
    end
    if best_cols === nothing
        error("Could not find full-rank set of columns over GF(2).")
    end
    return best_cols, best_score
end

# ========== Step 3: Row selection ==========

"""
    select_rows_gf2(G::Matrix{Int}, kept_cols::Vector{Int}, r::Int)

Select r rows to keep (remove k - r rows) so that resulting G' has full rank over GF(2)
and minimal number of 1s in the selected submatrix.
"""
function select_rows_gf2(G::Matrix{Int}, kept_cols::Vector{Int}, r::Int)
    k, _ = size(G)
    best_rows = nothing
    best_score = typemax(Int)
    for rows in combinations(1:k, r)
        Gsub = G[collect(rows), :]
        if gf2_rank(Gsub) == r
            score = sum(Gsub[:, kept_cols])
            if score < best_score
                best_score = score
                best_rows = collect(rows)
            end
        end
    end
    if best_rows === nothing
        error("Could not find suitable rows to keep.")
    end
    return best_rows, best_score
end

# ========== Step 4: Expand with rows of 2s ==========

"""
    construct_expanded_matrix_gf2(G::Matrix{Int})

Implements the algorithm over GF(2):
1. Compute rank r of G.
2. Select r columns (subset with full rank and minimal weight).
3. Remove k - r rows such that G' (r×n) still full rank.
4. Add (n - r) rows with a single 2 in the unselected columns.
Returns (G_expanded, info::Dict).
"""
function construct_expanded_matrix_gf2(G::Matrix{Int})
    k, n = size(G)
    r = gf2_rank(G)
    @assert r > 0 "Rank 0 matrix."

    cols, score_cols = select_columns_gf2(G, r)
    rows, score_rows = select_rows_gf2(G, cols, r)

    Gprime = G[rows, :]

    # Unselected columns
    unselected = setdiff(1:n, cols)

    # Add rows with a single 2
    extra = zeros(Int, n - r, n)
    for (i, c) in enumerate(unselected)
        extra[i, c] = 2
    end

    Gexp = vcat(Gprime, extra)

    info = Dict(
        :rank => r,
        :selected_cols => cols,
        :selected_rows => rows,
        :unselected_cols => unselected,
        :score_cols => score_cols,
        :score_rows => score_rows
    )
    return Gexp, info
end

# ========== Example ==========


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


Gexp, info = construct_expanded_matrix_gf2(G)
println("Original G:")
display(G)
println("\nExpanded G':")
det(Gexp)
println("\nInfo:\n", info)
