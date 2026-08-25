include("fixtures/ldlc_decoder_regression_utils.jl")

function test_ldlc_decoder_golden_regression()
    H = npzread(LDLC_REGRESSION_H_PATH)
    goldens = npzread(LDLC_REGRESSION_GOLDEN_PATH)
    schedules, algorithms = _ldlc_config_codes()

    @test only(goldens["version"]) == LDLC_REGRESSION_VERSION
    @test only(goldens["seed"]) == LDLC_REGRESSION_SEED
    @test only(goldens["sigma"]) == LDLC_REGRESSION_SIGMA
    @test only(goldens["iterations"]) == LDLC_REGRESSION_ITERATIONS
    @test only(goldens["golden_samples"]) == LDLC_REGRESSION_GOLDEN_SAMPLES
    @test vec(goldens["schedule_codes"]) == schedules
    @test vec(goldens["algorithm_codes"]) == algorithms

    noise = goldens["golden_noise"]
    soft_outputs = goldens["soft_outputs"]
    hard_decisions = goldens["hard_decisions"]

    @test size(H) == (128, 128)
    @test size(noise) == (128, LDLC_REGRESSION_GOLDEN_SAMPLES)
    @test size(soft_outputs) == (128, LDLC_REGRESSION_GOLDEN_SAMPLES, length(LDLC_REGRESSION_CONFIGS))
    @test size(hard_decisions) == (128, LDLC_REGRESSION_GOLDEN_SAMPLES, length(LDLC_REGRESSION_CONFIGS))

    for (config_idx, config) in enumerate(LDLC_REGRESSION_CONFIGS)
        @testset "$(config.schedule)-$(config.algorithm)" begin
            for sample_idx in 1:LDLC_REGRESSION_GOLDEN_SAMPLES
                bp, hd = _ldlc_decode(H, noise[:, sample_idx], config)

                @test length(bp) == 128
                @test all(isfinite, bp)
                @test hd == hard_decisions[:, sample_idx, config_idx]
                @test bp ≈ soft_outputs[:, sample_idx, config_idx] rtol = 1e-10 atol = 1e-10
            end
        end
    end
end
