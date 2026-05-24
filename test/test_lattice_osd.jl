using Test
using LatticeDecoder: generate_candidates, local_search!, LocalSearch, solution_weight
using LinearAlgebra
"""
    test_number_generated(n::Int, order::Vector{Int64})

Test if the number of generated candidates is as estimated analytically.
"""
function test_number_generated(n::Int, order::Vector{Int64})
    tot_expected = 0
    for i in eachindex(order)
        r = order[i]
        tot_expected += (2 * r)^i * binomial(n, i)
    end
    length(generate_candidates(n, order)) == tot_expected
end



"""
    test_no_dublicates(n::Int, order::Vector{Int64})

Test if there are no dublicates in the generated candidates.
"""
function test_no_dublicates(n::Int, order::Vector{Int64})
    vecs = generate_candidates(n, order)
    flag = true
    for i in eachindex(vecs)
        for j in i+1:length(vecs)
            if vecs[i] == vecs[j]
                flag = false
                break
            end
        end
    end
    flag
end

mutable struct TestCaseLatticeOSD
    H::AbstractMatrix{Float64}
    c::Vector{Float64}
    λ::Vector{Float64}
    G::AbstractMatrix{Float64}

    TestCaseLatticeOSD(H::AbstractMatrix{Float64}) = new(H, zeros(size(H, 1)), zeros(size(H, 1)), inv(H))

end


"""
    test_pauli_surface_correction(n::Int)

Test Pauli surface correction for the GKP code.
"""
function test_pauli_surface_correction(n::Int)
    return false
end

function brute_force_candidate_search(G::AbstractMatrix{Float64}, y::Vector{Float64}, dec::Vector{Int64}, order::Vector{Int64})
    r = y - G * dec
    best_u = zeros(Int64, size(G, 2))
    best_dist = sum(abs2, r)
    for candidate in generate_candidates(size(G, 2), order)
        dist = sum(abs2, r - G * candidate)
        if dist < best_dist
            best_dist = dist
            best_u .= candidate
        end
    end
    return best_u, best_dist
end

function test_local_search_full_enumeration_matches_bruteforce()
    G = [
        1.0 0.2 0.0
        0.0 1.0 0.3
        0.1 0.0 1.0
    ]
    y = G * [2, -1, 1] + [0.03, -0.02, 0.01]
    dec = zeros(Int64, 3)
    order = [2, 1, 1]

    expected_u, expected_dist = brute_force_candidate_search(G, y, copy(dec), order)
    lsd = LocalSearch(length(order), G, order; search=:full_enumeration)
    local_search!(y, zeros(3), dec, lsd)

    @test dec == expected_u
    @test sum(abs2, y - G * dec) ≈ expected_dist
end

function test_local_search_full_enumeration_searches_all_columns()
    G = Matrix{Float64}(I, 3, 3)
    y = [0.0, 0.0, 2.1]
    λ = [10.0, 1.0, 0.1]

    restricted_dec = zeros(Int64, 3)
    restricted = LocalSearch(1, G, [2], false, false, false)
    local_search!(y, λ, restricted_dec, restricted)

    full_enumeration_dec = zeros(Int64, 3)
    full_enumeration = LocalSearch(1, G, [2]; search=:full_enumeration)
    local_search!(y, λ, full_enumeration_dec, full_enumeration)

    @test restricted_dec == [0, 0, 0]
    @test full_enumeration_dec == [0, 0, 2]
    @test sum(abs2, y - G * full_enumeration_dec) < sum(abs2, y - G * restricted_dec)
end

function test_local_search_full_enumeration_never_worsens()
    G = [
        1.0 0.25
        0.1 1.0
    ]
    y = [0.2, -0.15]
    dec = [1, -1]
    initial_dist = sum(abs2, y - G * dec)

    lsd = LocalSearch(length([1, 1]), G, [1, 1]; search=:full_enumeration)
    local_search!(y, zeros(2), dec, lsd)

    @test sum(abs2, y - G * dec) <= initial_dist
end

function test_local_search_constructor_api()
    G = Matrix{Float64}(I, 3, 3)
    order = [1, 1]

    positional = LocalSearch(2, G, order, false, false, false)
    keyword = LocalSearch(2, G, order; search="restricted")
    full_enumeration = LocalSearch(2, G, order; search=:full_enumeration)
    sphere = LocalSearch(2, G, order; sphere_decoding=true)

    @test positional.search == :restricted
    @test keyword.search == :restricted
    @test full_enumeration.search == :full_enumeration
    @test length(positional.candidates[1]) == 2
    @test length(full_enumeration.candidates[1]) == size(G, 2)
    @test isempty(sphere.candidates)
    @test isempty(sphere.candidate_supports)
    @test_throws ArgumentError LocalSearch(2, G, order; search=:full_enumeration, sphere_decoding=true)
    @test_throws ArgumentError LocalSearch(2, G, order; search=:unknown)
end

function test_local_search_streaming_matches_materialized()
    G = [
        1.0 0.2 0.0
        0.0 1.0 0.3
        0.1 0.0 1.0
    ]
    y = G * [2, -1, 1] + [0.03, -0.02, 0.01]
    λ = [0.5, 0.4, 0.3]
    order = [2, 1, 1]

    materialized_dec = zeros(Int64, 3)
    materialized = LocalSearch(length(order), G, order)
    local_search!(y, λ, materialized_dec, materialized)

    streaming_dec = zeros(Int64, 3)
    streaming = LocalSearch(length(order), G, order; max_materialized_candidates=0)
    local_search!(y, λ, streaming_dec, streaming)

    @test !isempty(materialized.candidates)
    @test isempty(streaming.candidates)
    @test streaming_dec == materialized_dec
end

function test_local_search_large_orders_are_lazy()
    G = Matrix{Float64}(I, 12, 12)
    order = fill(1, 12)

    lazy = LocalSearch(length(order), G, order)
    sphere = LocalSearch(length(order), G, order; sphere_decoding=true)

    @test isempty(lazy.candidates)
    @test isempty(lazy.candidate_supports)
    @test isempty(sphere.candidates)
    @test isempty(sphere.candidate_supports)
end
