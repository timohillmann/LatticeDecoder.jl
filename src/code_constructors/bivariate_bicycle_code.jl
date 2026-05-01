using Distributed
# addprocs(16)  # add worker processes if not already added
@everywhere using StatsBase
@everywhere using NPZ
@everywhere using LinearAlgebra
@everywhere using Combinatorics
@everywhere using ProgressMeter
@everywhere using LinearAlgebraX
@everywhere using SparseArrays: sparse, SparseMatrixCSC
@everywhere using Random
# Id_block = (p)^(1/2) * I

# M = vcat((p)^(-1/2) * H, Id_block)

# J = symplectic_form(Int64(size(M)[2] / 2))

# A = M * J * transpose(M)


@everywhere const MAX_ITERS = 500_000

# """
#     sample_combinations_iter(r, k; max_samples=1000, rng=Random.GLOBAL_RNG)

# Return an iterator over combinations of `1:r` taken `k` at a time.

# - If the total number of combinations ≤ `max_samples`, it yields *all* combinations in random order.
# - Otherwise, it yields `max_samples` random combinations, without generating all combinations.
# """
# @everywhere function sample_combinations_iter(r, k; max_samples=1000, rng=Random.GLOBAL_RNG)
#     if is_small_enough(r, k, max_samples)
#         # all_combos = collect(combinations(1:r, k))
#         return combinations(1:r, k)
#     else
#         return RandomCombinationsIterator(r, k, max_samples, rng)
#     end
# end


# """
#     is_small_enough(r, k, max_samples)

# Checks if binomial(r, k) ≤ max_samples without overflow.
# """
# @everywhere function is_small_enough(r, k, max_samples)
#     k = min(k, r - k)
#     result = 1.0
#     for i in 1:k
#         result *= (r - i + 1) / i
#         if result > max_samples + 0.5
#             return false
#         end
#     end
#     return true
# end


# """
# An iterator that yields `max_samples` random k-combinations of 1:r.
# """
# @everywhere struct RandomCombinationsIterator
#     r::Int
#     k::Int
#     n::Int
#     rng::AbstractRNG
# end

# @everywhere Base.IteratorSize(::Type{RandomCombinationsIterator}) = Base.HasLength()
# @everywhere Base.length(it::RandomCombinationsIterator) = it.n

# @everywhere Base.iterate(it::RandomCombinationsIterator, state=1) =
#     state > it.n ? nothing : (random_combination(it.r, it.k, it.rng), state + 1)


# """
# Generate one random combination uniformly without enumerating all.
# """
# @everywhere function random_combination(r, k, rng)
#     result = Int[]
#     remaining = k
#     for i in 1:r
#         if remaining == 0
#             break
#         end
#         if rand(rng) < remaining / (r - i + 1)
#             push!(result, i)
#             remaining -= 1
#         end
#     end
#     return result
# end





# ========== Basic utilities ==========


@everywhere function modp_rank(M::AbstractMatrix{<:Integer}, p::Int)
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



@everywhere function build_full_rank_cols!(G::Matrix{Int}, p::Int, r::Int64, idx_set::Vector{Int64}, inds=Vector{Int64})
    k, n = size(G)
    randperm!(inds)
    crank = 1
    cG = G[:, inds[1]]
    idx_set[crank] = inds[1]
    @inbounds for k = 2:n
        S = hcat(cG, G[:, inds[k]])
        new_rank = modp_rank(S, p)
        if new_rank > crank
            crank = new_rank
            cG = S
            idx_set[crank] = inds[k]
        end
        if crank == r
            break
        end
    end
end


@everywhere function build_full_rank_rows!(G::AbstractMatrix{Int}, p::Int, r::Int64, idx_set::Vector{Int64}, inds::Vector{Int64})
    k, n = size(G)
    randperm!(inds)
    crank = 1
    cG = G[inds[1], :]
    idx_set[crank] = inds[1]
    @inbounds for kk = 2:k
        if crank == 1
            S = vcat(cG', G[inds[kk], :]')
        else
            S = vcat(cG, G[inds[kk], :]')
        end
        new_rank = modp_rank(S, p)
        if new_rank > crank
            crank = new_rank
            cG = S
            idx_set[crank] = inds[kk]
        end
        if crank == r
            break
        end
    end
end


# ========== Column selection ==========

# @everywhere function select_columns_modp(G::Matrix{Int}, r::Int, p::Int; max_iters::Int64=MAX_ITERS)
#     k, n = size(G)
#     best_cols = nothing
#     best_score = typemax(Int)
#     # prog = Progress(length(combinations(1:n, r)), desc="Selecting columns")
    
#     num_iter = 0

#     for cols in sample_combinations_iter(n, r; max_samples=max_iters) # combinations(1:n, r)
#         if num_iter % 10_000 == 0
#             # print("Iteration $(num_iter) \r")
#             # flush(stdout)
#         end
#         S = G[:, collect(cols)] .% p
#         println(cols)
#         flush(stdout)
#         if modp_rank(S, p) == r
#             score = det(S'*S)
#             if score < best_score
#                 best_score = score
#                 best_cols = collect(cols)
#             end
#         end
#         num_iter += 1
#         if num_iter > max_iters
#             best_cols === nothing && error("Cannot find full-rank columns mod $p")
#             return best_cols
#         end
#     end
#     best_cols === nothing && error("Cannot find full-rank columns mod $p")
#     return best_cols
# end

@everywhere function select_columns_modp(G::Matrix{Int}, r::Int, p::Int; max_iters::Int64=MAX_ITERS)
    k, n = size(G)
    best_cols = nothing
    best_score = Inf
    num_iter = 0
    idx_set = Vector{Int64}(undef, r)
    inds = Vector{Int64}(undef, n)
    for num_iter = 1:max_iters
        build_full_rank_cols!(G, p, r, idx_set, inds)
        score = det(G[:, idx_set]'*G[:, idx_set])
        if score < best_score
            best_score = score
            # println("Score: $(best_score)")
            best_cols = deepcopy(idx_set)
        end
    end
    return best_cols
end

# ========== Row selection ==========

# @everywhere function select_rows_modp(G::Matrix{Int}, kept_cols::Vector{Int}, r::Int, p::Int; max_iters::Int64=MAX_ITERS)
#     k, _ = size(G)
#     best_rows = nothing
#     best_score = typemax(Int)
#     num_iters = 0

#     # prog = Progress(length(combinations(1:k, r)), desc="Selecting rows")
#     for rows in sample_combinations_iter(k, r; max_samples=max_iters) # combinations(1:k, r)
#         # next!(prog)
#         if num_iters % 10_000 == 0
#             print("Iteration $(num_iters)  \r ")
#         end
#         Gsub = G[rows, :] .% p
#         if modp_rank(Gsub, p) == r
#             score = det(Gsub'* Gsub)
#             if score < best_score
#                 best_score = score
#                 best_rows = rows
#             end
#         end
#         num_iters += 1
#         if num_iters > max_iters
#             best_rows === nothing && error("Cannot find suitable rows mod $p")
#             return best_rows
#         end
#     end
#     best_rows === nothing && error("Cannot find suitable rows mod $p")
#     return best_rows
# end

@everywhere function select_rows_modp(G::AbstractMatrix{Int}, r::Int, p::Int; max_iters::Int64=MAX_ITERS)
    k, n = size(G)
    best_rows = nothing
    best_score = Inf
    num_iter = 0
    idx_set = Vector{Int64}(undef, r)
    inds = Vector{Int64}(undef, k)
    for num_iter = 1:max_iters
        # if num_iter % 10_000 == 0
        #     print("Iteration $(num_iter) \r")
        #     flush(stdout)
        # end
        build_full_rank_rows!(G, p, r, idx_set, inds)
        @views score = det(G[idx_set, :]'*G[idx_set, :])
        if score < best_score
            best_score = score
            # println("Score: $(best_score)")
            best_rows = deepcopy(idx_set)
        end
    end
    return best_rows
end
#     k, _ = size(G)
#     best_rows = nothing
#     best_score = typemax(Int)
#     num_iters = 0

#     # prog = Progress(length(combinations(1:k, r)), desc="Selecting rows")
#     for rows in sample_combinations_iter(k, r; max_samples=max_iters) # combinations(1:k, r)
#         # next!(prog)
#         if num_iters % 10_000 == 0
#             print("Iteration $(num_iters)  \r ")
#         end
#         Gsub = G[rows, :] .% p
#         if modp_rank(Gsub, p) == r
#             score = det(Gsub'* Gsub)
#             if score < best_score
#                 best_score = score
#                 best_rows = rows
#             end
#         end
#         num_iters += 1
#         if num_iters > max_iters
#             best_rows === nothing && error("Cannot find suitable rows mod $p")
#             return best_rows
#         end
#     end
#     best_rows === nothing && error("Cannot find suitable rows mod $p")
#     return best_rows

# ========== Main algorithm ==========


@everywhere function expand_matrix_modp(G::Matrix{Int}, p::Int)
    k, n = size(G)
    r = modp_rank(G, p)
    # println("Matrix rank: $(r)")
    @assert r > 0 "Matrix has rank 0 mod $p"

    cols = select_columns_modp(G, r, p)
    # println("Cols: ", cols)
    S = G[:, cols]
    rows = select_rows_modp(S, r, p)

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

# @everywhere function balance_weights!(mat::AbstractMatrix{Int64}; sign::Int64 = -1)
#     for i in 1:size(mat, 1)
#         row = mat[i, :]
#         num_nzvals = count(!iszero, row)
#         if num_nzvals == 1
#             nz_idx = row.nzind[1]
#             # find the row that has a 1 in the same column
#             for j in 1:size(mat, 1)
#                 if j == i
#                     continue
#                 end
#                 if mat[j, nz_idx] == 1
#                     row2 = mat[j, :]
#                     @. mat[i, :] = row2 + (sign * row)
#                     break
#                 end
#             end
#         end
#     end
# end


# @everywhere function randomize_signs!(M::AbstractMatrix; density::Float64=0.5)
#     n, m = size(M)
#     @assert n == m
#     T = typeof(M)

#     Dc = Diagonal(sample(T[-1, 1], Weights([density, 1-density]), n))
#     Dr = Diagonal(sample(T[-1, 1], Weights([density, 1-density]), n))

#     M .= Dr * M * Dc
#     return M
# end

# code_name = "108_8_10_p$(p)"
# PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name).npz";
# # code = npzread(PATH);





@everywhere function worker(code_name)
    println(code_name)
    p = 2
    w = 5
    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/weight_$(w)_p$(p)/$(code_name).npz";
    code = npzread(PATH);

    hx = code["hx"];
    hz = code["hz"];
    Z1 = zeros(size(hx, 1), size(hz, 2));
    Z2 = zeros(size(hz, 1), size(hx, 2));
    # H = [hx hz];

    Hx_exp, info = expand_matrix_modp(hx, p);
    Hz_exp, info = expand_matrix_modp(hz, p);


    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/weight_$(w)_p$(p)/$(code_name)_expanded.npz";
    code = npzwrite(PATH, Dict("hx" => Int.(Hx_exp), "hz" => Int.(Hz_exp), "lx" => code["lx"], "lz" => code["lz"]));
end

function main()
    # Read the file as a single string
    raw = open("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/weight_5_p2/out.txt") do f
        read(f, String)
    end

    # Split by comma, keep the quotes
    code_names = split(raw, ",")
    code_names = [cn[2:end-1] for cn in code_names]
    print(code_names)

    pmap(worker, code_names)
end

# main()

# # "40_2_8_p3_0","40_2_8_'p3_1","40_2_8_p3_2","48_2_8_p3_0","48_2_8_p3_1","48_2_8_p3_2","48_2_8_p3_3","48_2_8_p3_4","48_2_8_p3_5","48_2_8_p3_6","48_2_8_p3_7"
# # code_names = String["56_2_8_p3_0","56_2_8_p3_1","56_2_8_p3_2","60_2_8_p3_0","60_2_8_p3_1","60_2_8_p3_2","60_2_8_p3_3","60_2_8_p3_4","60_2_8_p3_5","60_2_8_p3_6","60_2_8_p3_7","60_2_8_p3_8","64_2_8_p3_0","64_2_8_p3_1","64_2_8_p3_10","64_2_8_p3_11","64_2_8_p3_12","64_2_8_p3_2","64_2_8_p3_3","64_2_8_p3_4","64_2_8_p3_5","64_2_8_p3_6","64_2_8_p3_7","64_2_8_p3_8","64_2_8_p3_9","64_8_4_p3_0","64_8_4_p3_1","72_2_8_p3_0","72_2_8_p3_1","72_8_6_p3_0","72_8_6_p3_1","80_2_8_p3_0","80_2_8_p3_1","80_2_8_p3_2","80_2_8_p3_3","80_2_8_p3_4","80_4_8_p3_0","80_8_5_p3_0","84_2_8_p3_0","84_2_8_p3_1","84_2_8_p3_2","84_2_8_p3_3","84_2_8_p3_4","84_2_8_p3_5","84_2_8_p3_6","88_2_8_p3_0","96_2_8_p3_0","96_2_8_p3_1","96_2_8_p3_10","96_2_8_p3_11","96_2_8_p3_12","96_2_8_p3_13","96_2_8_p3_14","96_2_8_p3_15","96_2_8_p3_2","96_2_8_p3_3","96_2_8_p3_4","96_2_8_p3_5","96_2_8_p3_6","96_2_8_p3_7","96_2_8_p3_8","96_2_8_p3_9","96_4_8_p3_0","96_4_8_p3_1","96_4_8_p3_2","96_4_8_p3_3","96_4_8_p3_4","96_8_5_p3_0"]
# # code_names = String["24_2_5_p3_0","32_2_4_p3_0","32_2_4_p3_1","32_2_4_p3_2","32_4_4_p3_0","32_4_4_p3_1","32_4_4_p3_2","36_2_6_p3_0","36_2_6_p3_1","36_2_6_p3_2","36_2_6_p3_3","36_2_6_p3_4","36_4_6_p3_0","36_4_6_p3_1","40_2_4_p3_0","40_2_4_p3_1","40_2_5_p3_0","40_2_5_p3_1","40_2_5_p3_2","40_2_5_p3_3","40_2_5_p3_4","40_2_5_p3_5","40_2_5_p3_6","40_2_5_p3_7","40_2_7_p3_0","40_2_7_p3_1","40_2_8_p3_0"]
# code_names = String["40_2_4_p3_0","40_2_4_p3_1","48_6_4_p3_0","48_6_4_p3_1","50_2_5_p3_0","50_2_5_p3_1","50_2_5_p3_2","56_2_4_p3_0","56_2_4_p3_1","60_2_4_p3_0","60_2_4_p3_1","60_4_4_p3_0","60_6_4_p3_0","60_6_5_p3_0","64_2_4_p3_0","64_2_4_p3_1","70_2_5_p3_0","70_2_5_p3_1","70_2_7_p3_0","70_2_7_p3_1","70_2_7_p3_2","72_12_4_p3_0","72_6_4_p3_0","72_6_4_p3_1","72_6_4_p3_2","80_2_5_p3_0","80_2_5_p3_1","80_2_8_p3_0","80_2_8_p3_1","80_2_8_p3_2","84_2_4_p3_0","84_4_4_p3_0","84_6_4_p3_0","84_6_5_p3_0","90_2_6_p3_0","90_6_5_p3_0","90_6_5_p3_1","90_6_5_p3_2","96_2_4_p3_0","96_4_4_p3_0","96_6_4_p3_0","96_6_5_p3_0","98_2_7_p3_0","98_2_7_p3_1"]
p = 2
for code_name in ["$(2*L^2)_2_$(L)_p2" for L=3:10]
    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/surface_codes/$(code_name).npz";
    code = npzread(PATH);

    hx = code["hx"];
    hz = code["hz"];
    Z1 = zeros(size(hx, 1), size(hz, 2));
    Z2 = zeros(size(hz, 1), size(hx, 2));
    # H = [hx hz];

    Hx_exp, info = expand_matrix_modp(hx, p);
    Hz_exp, info = expand_matrix_modp(hz, p);


    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/surface_codes/$(code_name)_expanded.npz";
    code = npzwrite(PATH, Dict("hx" => Int.(Hx_exp), "hz" => Int.(Hz_exp), "lx" => code["lx"], "lz" => code["lz"]));
end

# code_name = "36_2_7_p$(p)"
# PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_expanded.npz";
# code = npzread(PATH);

# hx = code["hx"] / sqrt(p);
# hz = code["hz"] / sqrt(p);

# det(hx)
# det(hz)

# G = hz
# k, n = size(G)
# r = modp_rank(G, p)
# println("Matrix rank: $(r)")
# @assert r > 0 "Matrix has rank 0 mod $p"

# inds = Vector{Int64}(undef, n)
# idx_set = Vector{Int64}(undef, r)

# build_full_rank_cols!(G, p, r, idx_set, inds)

# cols = select_columns_modp(G, r, p)
# println("Cols: ", cols)
# S = G[:, cols]
# rows = select_rows_modp(S, r, p)

# Gprime = G[rows, :]  # keep selected rows


# code_30_4_5 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/30_4_5_p2_expanded.npz")
# code_48_4_7 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/48_4_7_p2_expanded.npz")
# code_78_4_9 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/78_4_9_p2_expanded_v2.npz")
# code_18_4_3 = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/78_4_9_p2_expanded.npz")
# hx = code_30_4_5["hx"]
# hx = sparse(hx)
# balance_weights!(hx, sign=+1)

# # hx = Matrix(hx)
# for k = 1:size(hx, 1)
#     print(sum(hx[k, :] .!= 0), " ")
# # end
# code_names = ["40_2_4_p3_0","40_2_4_p3_1","48_6_4_p3_1","50_2_5_p3_0","50_2_5_p3_1","50_2_5_p3_2","56_2_4_p3_0","56_2_4_p3_1","60_2_4_p3_0","60_2_4_p3_1","60_4_4_p3_0"]
# # code_names = String["24_2_5_p3_0","32_2_4_p3_0","32_2_4_p3_1","32_2_4_p3_2","32_4_4_p3_0","32_4_4_p3_1","32_4_4_p3_2","36_2_6_p3_0","36_2_6_p3_1","36_2_6_p3_2","36_2_6_p3_3","36_2_6_p3_4","36_4_6_p3_0","36_4_6_p3_1","40_2_4_p3_0","40_2_4_p3_1","40_2_5_p3_0","40_2_5_p3_1","40_2_5_p3_2","40_2_5_p3_3","40_2_5_p3_4","40_2_5_p3_5","40_2_5_p3_6","40_2_5_p3_7","40_2_7_p3_0","40_2_7_p3_1","40_2_8_p3_0"]
# w = 6
# p = 3
# for code_name in code_names
#     PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/weight_$(w)_p$(p)/$(code_name)_expanded.npz";
#     code = npzread(PATH);
    
#     hx = code["hx"];
#     hz = code["hz"];
#     det_hx = abs((det(code["hx"] / sqrt(p))))
#     det_hz = abs((det(code["hz"] / sqrt(p))))
#     n = size(code["hx"], 1)
#     println("det_hx: $(det_hx)")
#     println("det_hz: $(det_hz)")
#     println(log(p, det_hx) + log(p, det_hz))

#     # println(n - log2(det_hx / BigInt(2^(n/2))) - log2(det_hz / BigInt(2^(n/2))))
# end



