using LinearAlgebra
using Nemo
using Test
using LatticeDecoder

function test_stack_gkp_generator()
    H = [
        1 1 0;
        0 1 1;
    ]
    generator = stack_gkp_generator(H)

    @test size(generator) == (3, 3)
    @test generator[1, :] == [2, 0, 0]
    @test generator[2:end, :] == H
    @test abs(round(Int, det(generator))) == 2

    dependent_H = [
        1 1 0;
        0 1 1;
        1 0 1;
    ]
    dependent_generator = stack_gkp_generator(dependent_H)
    @test size(dependent_generator) == (3, 3)
    @test abs(round(Int, det(dependent_generator))) == 2

    # Keeping the raw rows here would give determinant three despite the
    # matrix having full rank over GF(2).
    odd_determinant_H = ones(Int, 4, 4) - Matrix{Int}(I, 4, 4)
    exact_volume_generator = stack_gkp_generator(odd_determinant_H)
    @test abs(round(Int, det(exact_volume_generator))) == 1

    zero_generator = stack_gkp_generator(zeros(Int, 2, 3))
    @test zero_generator == 2Matrix{Int}(I, 3, 3)

    Z2, _ = residue_ring(ZZ, 2)
    nemo_H = matrix(Z2, H)
    @test stack_gkp_generator(nemo_H) == generator

    Z3, _ = residue_ring(ZZ, 3)
    @test_throws ArgumentError stack_gkp_generator(matrix(Z3, H))
end

function test_reduced_gkp_repetition_code()
    for distance in (2, 3, 5), bit_flip in (false, true)
        code = GKP_Rep_Code(distance, bit_flip, true)
        @test size(code.code) == (2distance, 2distance)
        @test all(isfinite, code.code)
        @test !iszero(det(code.code))
    end
end

function test_reduced_gkp_repetition_code_stays_sparse()
    distance = 7
    code = GKP_Rep_Code(distance, false, true)
    sector = code.code[(distance + 1):end, (distance + 1):end]

    @test maximum(vec(sum(!iszero, sector; dims=1))) == 2
    @test maximum(vec(sum(!iszero, sector; dims=2))) == 2
end
