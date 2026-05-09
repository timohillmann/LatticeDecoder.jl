using Test

include("../fixtures/ldlc_decoder_regression_utils.jl")

function test_ldlc_decoder_statistics_regression()
    H = npzread(LDLC_REGRESSION_H_PATH)
    goldens = npzread(LDLC_REGRESSION_GOLDEN_PATH)

    @test only(goldens["version"]) == LDLC_REGRESSION_VERSION
    @test only(goldens["stat_samples"]) == LDLC_REGRESSION_STAT_SAMPLES

    noise = _ldlc_sample_noise(size(H, 1), LDLC_REGRESSION_STAT_SAMPLES)
    symbol_errors, totals, frame_failures, means, histogram = _ldlc_statistics(H, noise)

    @test symbol_errors == goldens["stat_symbol_errors"]
    @test totals == vec(goldens["stat_total_symbol_errors"])
    @test frame_failures == vec(goldens["stat_frame_failures"])
    @test means ≈ vec(goldens["stat_mean_symbol_errors"]) rtol = 0 atol = 0
    @test histogram == goldens["stat_histogram"]
end

@testset "LDLCDecoder statistics regression" begin
    test_ldlc_decoder_statistics_regression()
end
