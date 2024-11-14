using LinearAlgebra
using Combinatorics
using ResumableFunctions
using LLLplus

struct LatticeStatisticsDecoding
    order::Vector{Int64}
    G::AbstractMatrix{Float64}
    B::AbstractMatrix{Float64}
    candidates::Vector{Vector{Int64}}
    basis::AbstractMatrix{Float64}
end



"""
    LatticeStatisticsDecoding(order::Vector{Int64}, G::AbstractMatrix{Float64}, reduced::Bool = true)

Create a new instance of the LatticeStatisticsDecoding algorithm. The algorithm is used to solve the lattice decoding problem. The algorithm uses the LLL basis reduction algorithm to reduce the basis of the lattice. The algorithm generates all possible candidates for the solution of the lattice decoding problem.
"""
function LatticeStatisticsDecoding(order::Vector{Int64}, G::AbstractMatrix{Float64}, reduced::Bool=true)
    candidates = generate_candidates(size(G, 2), order)

    if reduced
        B = lll_basis_reduction(G)
        basis = lll_basis_reduction(G)
    else
        B = G
        basis = G
    end

    return LatticeStatisticsDecoding(order, G, B, candidates, basis)

end


function lll_basis_reduction(G::AbstractMatrix{Float64})
    B, _ = LLLplus.lll(G)
    return B
end


"""
    lattice_statistics_decoding!(c::Vector{Float64}, λ::Vector{Float64}, lsd::LatticeStatisticsDecoding)

Perform the lattice statistics decoding algorithm. The algorithm checks whether the current solution `c` can be improved by adding a candidate vector to it. The candidate is accepted if the weight of the new solution is smaller than the weight of the current solution. In principle the solution weight function can be any function that measures the quality of the solution.

"""
function lattice_statistics_decoding!(c::Vector{Float64}, λ::Vector{Float64}, lsd::LatticeStatisticsDecoding)
    w = solution_weight(c)
    for candidate in lsd.candidates
        _c = c - lsd.basis * candidate
        w_candidate = solution_weight(_c, λ)
        if w_candidate < w
            c .= _c
            w = w_candidate
        end
    end
end


function local_search!(c::Vector{Float64}, η::Vector{Float64}, dec::Vector{Int64}, λ::Vector{Float64}, lsd::LatticeStatisticsDecoding)
    _dec = copy(dec)
    w = solution_weight(η - lsd.basis * dec, λ)
    for candidate in lsd.candidates
        _dec = dec + candidate
        _c = η - lsd.basis * _dec
        w_candidate = solution_weight(_c)
        if w_candidate < w
            c = copy(_c)
            w = copy(w_candidate)
        end
    end
end

local_search!(c::Vector{Float64}, η::Vector{Float64}, dec::Vector{Int64}, lsd::LatticeStatisticsDecoding) = local_search!(c, η, dec, η, lsd)


lattice_statistics_decoding!(c::Vector{Float64}, lsd::LatticeStatisticsDecoding) = lattice_statistics_decoding!(c, c, lsd)

function solution_weight(b::Vector{Float64})
    return norm(b)
end


function solution_weight(b::Vector{Float64}, s::Float64)
    return norm(b .% s)
end

solution_weight(b::Vector{Float64}, λ::Vector{Float64}) = solution_weight(b)



"""
    generate_candidates(n::Int64, order::Vector{Int64})

Generate all possible candidates for the solution of the lattice decoding problem.
"""
function generate_candidates(n::Int64, order::Vector{Int64})
    vecs = []
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


function generate_candidates(n::Int64, order::Vector{Int64}, fault_indices::Vector{Int64})
    vecs = []
    _vec = zeros(Int64, n)
    for i in eachindex(order)
        r = order[i]
        rs = [s * _r for _r in 1:r if r != 0 for s in [-1, 1]]
        # printstyled("rs: $rs\n", color=:red)
        for r in generate_value_sets(rs, i)  # multiset_combinations(rs, i)
            # i could change the range 1:n to contain only a subset of the indices based on soft information
            # printstyled("r: $r\n", color=:blue)
            for idx in multiset_combinations(fault_indices, i)
                # printstyled("idx: $idx\n", color=:green)
                _vec .= 0
                _vec[idx] .= r
                push!(vecs, copy(_vec))
            end
        end
    end

    return vecs
end


@resumable function generate_candidates_iterator(n::Int64, order::Vector{Int64}, fault_indices::Vector{Int64})
    _vec = zeros(Int64, n)
    for i in eachindex(order)
        r = order[i]
        rs = [s * _r for _r in 1:r if r != 0 for s in [-1, 1]]
        # printstyled("rs: $rs\n", color=:red)
        for r in generate_value_sets(rs, i)  # multiset_combinations(rs, i)
            # i could change the range 1:n to contain only a subset of the indices based on soft information
            # printstyled("r: $r\n", color=:blue)
            for idx in multiset_combinations(fault_indices, i)
                # printstyled("idx: $idx\n", color=:green)
                _vec .= 0
                _vec[idx] .= r
                @yield copy(_vec)
            end
        end
    end
end


@resumable function generate_candidates_iterator(n::Int64, order::Vector{Int64})
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
                @yield r # copy(_vec)
            end
        end
    end
end



# it = generate_candidates_iterator(256, [5, 2, 1], collect(1:20))

"""
    generate_value_sets(rs::Vector{Int64}, i::Int64)

Generate all possible sets of length `i` from the values in `rs`.

"""
function generate_value_sets(rs::Vector{Int64}, i::Int64)
    return Iterators.product(fill(rs, (1, i))...)
end


"""
    generate_fault_indices(λ::Vector{Float64}, threshold::Float64)

Generate the indices of the faults based on a threshold value.
"""
function generate_fault_indices(λ::Vector{Float64}, threshold::Float64)
    return findall(λ .> threshold)
end


"""
    generate_fault_indices(λ::Vector{Float64}, n_faults::Int64)

Generate the fault indices based on the `n_faults` largest values in `λ`.
"""
function generate_fault_indices(λ::Vector{Float64}, n_faults::Int64)
    sort_idx = sortperm(λ, rev=true)
    return sort_idx[1:n_faults]
end



# H = [[1, 0, 0, 0, 0] [1 / 2, 1 / 2, 0, 0, 0] [0, 1 / 2, 1 / 2, 0, 0] [0, 0, 1 / 2, 1 / 2, 0] [0, 0, 0, 1 / 2, 1 / 2]]
# G = inv(H)
# n = 5
# y = 0.1 * randn(n)
# b = [0, 0, 0, 0, 0]
# λ = abs.(y - G * b)

# sort_idx = sortperm(λ, rev=true)



# using BenchmarkTools

# @benchmark generate_candidates(256, [5, 2, 1], collect(1:20))

# @benchmark for _ in generate_candidates_iterator(256, [5, 2, 1], collect(1:20))
# end


# @resumable function generate_value_sets_iterator(rs::Vector{Int64}, i::Int64)
#     let i = i
#         for r in generate_value_sets(rs, i)
#             @yield r
#         end
#     end
# end


# @resumable function generate_candidates_iterator(n::Int64, order::Vector{Int64})
#     _vec = zeros(Int64, n)
#     for i in eachindex(order)
#         let i = i
#             r = order[i]
#             rs = [s * _r for _r in 1:r if r != 0 for s in [-1, 1]]
#             let rs = rs
#                 for r in generate_value_sets_iterator(rs, i)
#                     @yield r
#                 end
#             end
#         end
#         #     # printstyled("rs: $rs\n", color=:red)
#         #         # i could change the range 1:n to contain only a subset of the indices based on soft information
#         #         # printstyled("r: $r\n", color=:blue)
#         #         for idx in multiset_combinations(1:n, i)
#         #             # printstyled("idx: $idx\n", color=:green)
#         #             _vec .= 0
#         #             _vec[idx] .= r
#         #             @yield r # copy(_vec)
#         #         end
#         # end
#     end
# end

# iter = generate_candidates_iterator(25, [5, 2, 1])

# for x in iter
#     println(x)
# end


# function generate_candidates(n::Int64, order::Vector{Int64}, fault_indices::Vector{Int64})
#     return Channel{Vector{Int64}}(Inf) do ch
#         _vec = zeros(Int64, n)
#         for i in eachindex(order)
#             r = order[i]
#             rs = [s * _r for _r in 1:r if r != 0 for s in [-1, 1]]
#             for r_set in generate_value_sets(rs, i)
#                 for idx in multiset_combinations(fault_indices, i)
#                     _vec .= 0
#                     _vec[idx] .= r_set
#                     put!(ch, copy(_vec))
#                 end
#             end
#         end
#     end
# end



# @noinline function test_channel(n::Int, order::Vector{Int}, fault_indices::Vector{Int})
#     ch = generate_candidates(n, order, fault_indices)
#     v = zeros(Int, n)
#     for _c in ch
#         v = _c
#     end
#     # v
# end

# @benchmark test_channel(256, [5, 2, 1], collect(1:20))
# @benchmark generate_candidates(256, [5, 2, 1], collect(1:20))