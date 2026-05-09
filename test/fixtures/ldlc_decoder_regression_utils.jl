using LatticeDecoder
using NPZ
using Random
using Statistics

const LDLC_REGRESSION_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const LDLC_REGRESSION_H_PATH = joinpath(
    LDLC_REGRESSION_ROOT,
    "generator_matrices",
    "classical_ldlc",
    "ldlc_n128_d5_H.npy",
)
const LDLC_REGRESSION_GOLDEN_PATH = joinpath(@__DIR__, "ldlc_n128_d5_decoder_goldens.npz")

const LDLC_REGRESSION_SEED = 20260509
const LDLC_REGRESSION_SIGMA = 0.20
const LDLC_REGRESSION_ITERATIONS = 5
const LDLC_REGRESSION_GOLDEN_SAMPLES = 4
const LDLC_REGRESSION_STAT_SAMPLES = 256
const LDLC_REGRESSION_VERSION = 1

const LDLC_REGRESSION_CONFIGS = (
    (schedule=:parallel, algorithm=:lsd),
    (schedule=:parallel, algorithm=:nearest),
    (schedule=:serial, algorithm=:lsd),
    (schedule=:serial, algorithm=:nearest),
)

const LDLC_REGRESSION_SCHEDULE_CODES = Dict(:parallel => 1, :serial => 2)
const LDLC_REGRESSION_ALGORITHM_CODES = Dict(:lsd => 1, :nearest => 2)

function _ldlc_config_codes()
    schedules = [LDLC_REGRESSION_SCHEDULE_CODES[config.schedule] for config in LDLC_REGRESSION_CONFIGS]
    algorithms = [LDLC_REGRESSION_ALGORITHM_CODES[config.algorithm] for config in LDLC_REGRESSION_CONFIGS]
    return schedules, algorithms
end

function _ldlc_decode(H, y, config)
    decoder = LDLCDecoder(
        initialize_tanner_graph(H);
        schedule=config.schedule,
        algorithm=config.algorithm,
        sigma=LDLC_REGRESSION_SIGMA,
        max_iterations=LDLC_REGRESSION_ITERATIONS,
    )
    bp = copy(run_decoder!(decoder, y))
    hd = hard_decision(bp, H)
    return bp, hd
end

function _ldlc_sample_noise(n::Integer, samples::Integer; seed::Integer=LDLC_REGRESSION_SEED)
    Random.seed!(seed)
    noise = Matrix{Float64}(undef, n, samples)
    for sample_idx in 1:samples
        noise[:, sample_idx] .= sample_error(LDLC_REGRESSION_SIGMA, n)
    end
    return noise
end

function _ldlc_statistics(H, noise)
    nconfigs = length(LDLC_REGRESSION_CONFIGS)
    nsamples = size(noise, 2)
    symbol_errors = Matrix{Int64}(undef, nconfigs, nsamples)

    for (config_idx, config) in enumerate(LDLC_REGRESSION_CONFIGS)
        for sample_idx in 1:nsamples
            bp, _ = _ldlc_decode(H, noise[:, sample_idx], config)
            symbol_errors[config_idx, sample_idx] = count_symbol_errors(hard_decision(bp, H))
        end
    end

    totals = vec(sum(symbol_errors; dims=2))
    frame_failures = vec(sum(symbol_errors .!= 0; dims=2))
    means = vec(mean(symbol_errors; dims=2))
    max_error = maximum(symbol_errors)
    histogram = zeros(Int64, nconfigs, max_error + 1)

    for config_idx in 1:nconfigs
        for value in symbol_errors[config_idx, :]
            histogram[config_idx, value + 1] += 1
        end
    end

    return symbol_errors, totals, frame_failures, means, histogram
end
