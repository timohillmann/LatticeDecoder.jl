using LinearAlgebra
using Random
using Test
using LatticeDecoder

function _identity_decoder(n; sigma=0.1)
    H = Matrix{Float64}(I, n, n)
    return LDLCDecoder(
        initialize_tanner_graph(H);
        schedule=:parallel,
        algorithm=:nearest,
        sigma,
        max_iterations=1,
    )
end

function test_binomial_estimate()
    estimate = BinomialEstimate(2, 10, 5)
    @test estimate.events == 2
    @test estimate.trials == 10
    @test estimate.samples == 5
    @test estimate.rate == 0.2
    @test 0 <= estimate.lower <= estimate.rate <= estimate.upper <= 1

    zero_estimate = BinomialEstimate(0, 10, 10)
    full_estimate = BinomialEstimate(10, 10, 10)
    @test zero_estimate.lower == 0
    @test full_estimate.upper == 1
    @test_throws ArgumentError BinomialEstimate(-1, 10, 10)
    @test_throws ArgumentError BinomialEstimate(11, 10, 10)
    @test_throws ArgumentError BinomialEstimate(0, 0, 0)
end

function test_classical_simulation_reproducibility()
    H = Matrix{Float64}(I, 3, 3)
    problem = ClassicalDecodingProblem(H)

    first_estimate = estimate_symbol_error_rate!(
        MersenneTwister(2026),
        _identity_decoder(3),
        problem;
        samples=12,
    )
    second_estimate = estimate_symbol_error_rate!(
        MersenneTwister(2026),
        _identity_decoder(3),
        problem;
        samples=12,
    )

    @test first_estimate == second_estimate
    @test first_estimate.samples == 12
    @test first_estimate.trials == 36
end

function test_quantum_simulation_reproducibility()
    H = Matrix{Float64}(I, 2, 2)
    problem = QuantumDecodingProblem(H, H, H)

    received = estimate_logical_error_rate!(
        MersenneTwister(7),
        _identity_decoder(2),
        problem;
        samples=10,
    )
    syndrome = estimate_logical_error_rate!(
        MersenneTwister(7),
        _identity_decoder(2),
        problem;
        samples=10,
        representative=:syndrome,
    )
    local_search = LocalSearch(2, H, [1])
    searched = estimate_logical_error_rate!(
        MersenneTwister(7),
        _identity_decoder(2),
        problem;
        samples=3,
        local_search,
    )

    @test received.events == 0
    @test syndrome.events == 0
    @test searched.events == 0
    @test !is_logical_error(H, [1.0, -2.0])
    @test is_logical_error(H, [0.5, 0.0])
end

function test_simulation_validation()
    H = Matrix{Float64}(I, 2, 2)
    classical = ClassicalDecodingProblem(H)
    quantum = QuantumDecodingProblem(H, H, H)

    @test_throws DimensionMismatch ClassicalDecodingProblem(ones(2, 3), ones(3, 2))
    @test_throws DimensionMismatch QuantumDecodingProblem(ones(2, 3), ones(2, 2), H)
    @test_throws ArgumentError estimate_symbol_error_rate!(
        MersenneTwister(1),
        _identity_decoder(2),
        classical;
        samples=0,
    )
    @test_throws ArgumentError estimate_logical_error_rate!(
        MersenneTwister(1),
        _identity_decoder(2),
        quantum;
        samples=1,
        representative=:invalid,
    )
    @test_throws DimensionMismatch estimate_symbol_error_rate!(
        MersenneTwister(1),
        _identity_decoder(3),
        classical;
        samples=1,
    )
end
