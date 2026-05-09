include("ldlc_decoder_regression_utils.jl")

function generate_ldlc_decoder_goldens(; output_path=LDLC_REGRESSION_GOLDEN_PATH)
    H = npzread(LDLC_REGRESSION_H_PATH)
    n = size(H, 1)
    nconfigs = length(LDLC_REGRESSION_CONFIGS)

    golden_noise = _ldlc_sample_noise(n, LDLC_REGRESSION_GOLDEN_SAMPLES)
    soft_outputs = Array{Float64}(undef, n, LDLC_REGRESSION_GOLDEN_SAMPLES, nconfigs)
    hard_decisions = Array{Int64}(undef, n, LDLC_REGRESSION_GOLDEN_SAMPLES, nconfigs)

    for (config_idx, config) in enumerate(LDLC_REGRESSION_CONFIGS)
        for sample_idx in 1:LDLC_REGRESSION_GOLDEN_SAMPLES
            bp, hd = _ldlc_decode(H, golden_noise[:, sample_idx], config)
            soft_outputs[:, sample_idx, config_idx] .= bp
            hard_decisions[:, sample_idx, config_idx] .= hd
        end
    end

    stat_noise = _ldlc_sample_noise(n, LDLC_REGRESSION_STAT_SAMPLES)
    stat_symbol_errors, stat_totals, stat_frame_failures, stat_means, stat_histogram =
        _ldlc_statistics(H, stat_noise)
    schedules, algorithms = _ldlc_config_codes()

    npzwrite(
        output_path,
        Dict(
            "version" => [LDLC_REGRESSION_VERSION],
            "seed" => [LDLC_REGRESSION_SEED],
            "sigma" => [LDLC_REGRESSION_SIGMA],
            "iterations" => [LDLC_REGRESSION_ITERATIONS],
            "golden_samples" => [LDLC_REGRESSION_GOLDEN_SAMPLES],
            "stat_samples" => [LDLC_REGRESSION_STAT_SAMPLES],
            "schedule_codes" => schedules,
            "algorithm_codes" => algorithms,
            "golden_noise" => golden_noise,
            "soft_outputs" => soft_outputs,
            "hard_decisions" => hard_decisions,
            "stat_symbol_errors" => stat_symbol_errors,
            "stat_total_symbol_errors" => stat_totals,
            "stat_frame_failures" => stat_frame_failures,
            "stat_mean_symbol_errors" => stat_means,
            "stat_histogram" => stat_histogram,
        ),
    )

    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    path = generate_ldlc_decoder_goldens()
    println("Wrote LDLC decoder golden fixture to $(path)")
end
