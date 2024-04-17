using Test
include("test_gaussians.jl")
include("test_code_reduction.jl")
include("test_lsd_decoder.jl")

# run test set
@testset "gaussians.jl" begin
    test_gaussian_assignment()
    test_gaussian_divide()
    test_moment_matching()
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
end;