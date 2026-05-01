using Test
include("test_lsd_paper_search.jl")
include("test_gaussians.jl")
include("test_code_reduction.jl")
include("test_lsd_decoder.jl")
include("test_lattice_osd.jl")

# run test set
@testset "gaussians.jl" begin
    test_gaussian_assignment()
    test_gaussian_divide()
    test_moment_matching()
    test_moment_matching!()
end;

@testset "concatenated_code_reduction.jl" begin
    toric_4d_test_set()
    toric_4d_test_set()
end;

@testset "lsd_decoder.jl" begin
    TC1 = test_case_one()
    test_collect_msg_vector(TC1)
    test_all_t_vectors(TC1)
    test_simplified_lsd(TC1)

    TC2 = test_case_two()
    test_collect_msg_vector(TC2)
    test_all_t_vectors(TC2)
    test_simplified_lsd(TC2)
end;


@testset "correct_number" begin
    for n in 3:10
        for w in 1:3
            for order in [[0, 0, w], [0, w, 0], [w, 0, 0], [w, w - 1, 1]]
                @test test_number_generated(n, order)
            end
        end
    end
end


@testset "test_pauli_rep_correction" begin
    for n in 3:2:19
        test_pauli_rep_correction(n)
    end
end
