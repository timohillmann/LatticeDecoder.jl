include("test_gaussians.jl")
include("test_code_reduction.jl")

# run test set
@testset "gaussians.jl" begin
    test_gaussian_assignment()
    test_gaussian_divide()
end

@testset "concatenated_code_reduction.jl" begin
    toric_4d_test_set()
    toric_4d_test_set()
end