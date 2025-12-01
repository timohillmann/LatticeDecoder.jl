using NPZ
using LinearAlgebra
using Combinatorics
using ProgressMeter
using LinearAlgebraX
using SparseArrays: sparse, SparseMatrixCSC
using Random
# Id_block = (p)^(1/2) * I

# M = vcat((p)^(-1/2) * H, Id_block)

# J = symplectic_form(Int64(size(M)[2] / 2))

# A = M * J * transpose(M)


const MAX_ITERS = 100_000_000

"""
    sample_combinations_iter(r, k; max_samples=1000, rng=Random.GLOBAL_RNG)

Return an iterator over combinations of `1:r` taken `k` at a time.

- If the total number of combinations ≤ `max_samples`, it yields *all* combinations in random order.
- Otherwise, it yields `max_samples` random combinations, without generating all combinations.
"""
function sample_combinations_iter(r, k; max_samples=1000, rng=Random.GLOBAL_RNG)
    if is_small_enough(r, k, max_samples)
        # all_combos = collect(combinations(1:r, k))
        return combinations(1:r, k)
    else
        return RandomCombinationsIterator(r, k, max_samples, rng)
    end
end


"""
    is_small_enough(r, k, max_samples)

Checks if binomial(r, k) ≤ max_samples without overflow.
"""
function is_small_enough(r, k, max_samples)
    k = min(k, r - k)
    result = 1.0
    for i in 1:k
        result *= (r - i + 1) / i
        if result > max_samples + 0.5
            return false
        end
    end
    return true
end


"""
An iterator that yields `max_samples` random k-combinations of 1:r.
"""
struct RandomCombinationsIterator
    r::Int
    k::Int
    n::Int
    rng::AbstractRNG
end

Base.IteratorSize(::Type{RandomCombinationsIterator}) = Base.HasLength()
Base.length(it::RandomCombinationsIterator) = it.n

Base.iterate(it::RandomCombinationsIterator, state=1) =
    state > it.n ? nothing : (random_combination(it.r, it.k, it.rng), state + 1)


"""
Generate one random combination uniformly without enumerating all.
"""
function random_combination(r, k, rng)
    result = Int[]
    remaining = k
    for i in 1:r
        if remaining == 0
            break
        end
        if rand(rng) < remaining / (r - i + 1)
            push!(result, i)
            remaining -= 1
        end
    end
    return result
end





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

function select_columns_modp(G::Matrix{Int}, r::Int, p::Int; max_iters::Int64=MAX_ITERS)
    k, n = size(G)
    best_cols = nothing
    best_score = typemax(Int)
    # prog = Progress(length(combinations(1:n, r)), desc="Selecting columns")
    
    num_iter = 0

    for cols in sample_combinations_iter(n, r; max_samples=max_iters) # combinations(1:n, r)
        if num_iter % 10_000 == 0
            print("Iteration $(num_iter) \r")
            flush(stdout)
        end
        S = G[:, collect(cols)] .% p
        if modp_rank(S, p) == r
            score = det(S'*S)
            if score < best_score
                best_score = score
                best_cols = collect(cols)
            end
        end
        num_iter += 1
        if num_iter > max_iters
            best_cols === nothing && error("Cannot find full-rank columns mod $p")
            return best_cols
        end
    end
    best_cols === nothing && error("Cannot find full-rank columns mod $p")
    return best_cols
end

# ========== Row selection ==========

function select_rows_modp(G::Matrix{Int}, kept_cols::Vector{Int}, r::Int, p::Int; max_iters::Int64=MAX_ITERS)
    k, _ = size(G)
    best_rows = nothing
    best_score = typemax(Int)
    num_iters = 0

    # prog = Progress(length(combinations(1:k, r)), desc="Selecting rows")
    for rows in sample_combinations_iter(k, r; max_samples=max_iters) # combinations(1:k, r)
        # next!(prog)
        if num_iters % 10_000 == 0
            print("Iteration $(num_iters)  \r ")
            flush(stdout)
        end
        Gsub = G[rows, :] .% p
        if modp_rank(Gsub, p) == r
            score = det(Gsub'* Gsub)
            if score < best_score
                best_score = score
                best_rows = rows
            end
        end
        num_iters += 1
        if num_iters > max_iters
            best_rows === nothing && error("Cannot find suitable rows mod $p")
            return best_rows
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

function balance_weights!(mat::AbstractMatrix{Int64})
    for i in 1:size(mat, 1)
        row = mat[i, :]
        num_nzvals = count(!iszero, row)
        if num_nzvals == 1
            nz_idx = row.nzind[1]
            # find the row that has a 1 in the same column
            for j in 1:size(mat, 1)
                if j == i
                    continue
                end
                if mat[j, nz_idx] == 1
                    row2 = mat[j, :]
                    mat[i, :] .= row2 .- row
                    break
                end
            end
        end
    end
end

p = 2
code_name = "18_4_3_p$(p)"
# PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name).npz";
# # code = npzread(PATH);


for code_name in ["18_4_3_p$(p)"]
    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name).npz";
    code = npzread(PATH);

    hx = code["hx"];
    hz = code["hz"];
    Z1 = zeros(size(hx, 1), size(hz, 2));
    Z2 = zeros(size(hz, 1), size(hx, 2));
    # H = [hx hz];

    Hx_exp, info = expand_matrix_modp(hx, p);
    Hz_exp, info = expand_matrix_modp(hz, p);


    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_expanded.npz";
    code = npzwrite(PATH, Dict("hx" => Int.(Hx_exp), "hz" => Int.(Hz_exp), "lx" => code["lx"], "lz" => code["lz"]));
end

# code_30_4_5 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/30_4_5_p2_expanded.npz")
# code_48_4_7 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/48_4_7_p2_expanded.npz")
# code_78_4_9 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/78_4_9_p2_expanded_v2.npz")
code_18_4_3 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/78_4_9_p2_expanded.npz")
# hx = code_30_4_5["hx"]
# hx = sparse(hx)
# balance_weights!(hx)


p = 2
for code in [code_18_4_3]
    det_hx = abs((det(code["hx"] / sqrt(2))))
    det_hz = abs((det(code["hz"] / sqrt(2))))
    n = size(code["hx"], 1)
    println(log2(det_hx) + log2(det_hz))

    # println(n - log2(det_hx / BigInt(2^(n/2))) - log2(det_hz / BigInt(2^(n/2))))
end

