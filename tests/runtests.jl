include("test_gaussians.jl")

# run test set
@testset "gaussians.jl" begin
    test_gaussian_assignment()
    test_gaussian_divide()
end