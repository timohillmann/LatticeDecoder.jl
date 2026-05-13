using Test

module TestLSDPaperSearch

using Test
using LatticeDecoder

struct QuantumCode
    code
end

include(joinpath(@__DIR__, "..", "src", "bp_algorithms", "tanner_graph_log_weight.jl"))
include(joinpath(@__DIR__, "..", "src", "bp_algorithms", "list_sphere_decoder_log_weight.jl"))

function zero_log_weight_inputs(n::Int)
    msgs = [gaussian_log_weight(0.0, 0.04, 0.0, 1.0) for _ in 1:n]
    inputs = ListSphereDecodingInput(msgs)
    inputs.β = 0.1
    inputs.β1 = 0.1
    return inputs
end

@testset "paper LSD search" begin
    inputs = zero_log_weight_inputs(53)

    old_L, old_D = simplified_lsd_legacy(inputs)
    @test !isempty(old_L)
    @test !isempty(old_D)

    inputs = zero_log_weight_inputs(53)
    result = simplified_lsd_paper(inputs)
    @test result.status == :ok
    @test !isempty(result.L)
    @test !isempty(result.D)
    @test minimum(result.D) ≈ 0.0
    @test all(result.D .<= inputs.β^2)
    @test result.visits < LatticeDecoder.MAX_ITER
    @test length(old_D) > length(result.D)

    default_L, default_D = simplified_lsd(inputs)
    @test default_L == result.L
    @test default_D == result.D

    budgeted = simplified_lsd_paper(inputs; max_visits=1)
    @test budgeted.status == :budget_exhausted
    @test budgeted.visits == 1
end

end
