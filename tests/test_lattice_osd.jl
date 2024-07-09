using Test
using LatticeDecoder: generate_candidates, solution_weight, LatticeStatisticsDecoding, lattice_statistics_decoding!, GKP_Rep_Code
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
    test_pauli_rep_correction(n::Int)

Test the lattice statistics decoding algorithm for the repetion code.
"""
function test_pauli_rep_correction(n::Int)
    gkp_code = GKP_Rep_Code(n, false, true)
    H = gkp_code.code[(n+1):end, (n+1):end]
    L = gkp_code.logical[1:n]
    RepCode = TestCaseLatticeOSD(H)
    order = zeros(Int64, n)
    order[n] = 1
    LSD = LatticeStatisticsDecoding(order, RepCode.G)
    lattice_statistics_decoding!(L, RepCode.λ, LSD)

    isapprox(norm(L), 0.0, atol=1e-8) == true
end


"""
    test_pauli_surface_correction(n::Int)

Test Pauli surface correction for the GKP code.
"""
function test_pauli_surface_correction(n::Int)
    return false
end

