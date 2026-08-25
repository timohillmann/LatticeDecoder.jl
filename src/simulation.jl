using LinearAlgebra
using Random

"""
    ClassicalDecodingProblem(H, G=inv(H))

A classical lattice decoding problem. `H` maps a decoder estimate to integer
symbols and `G` maps symbols to lattice points.
"""
struct ClassicalDecodingProblem{TH<:AbstractMatrix,TG<:AbstractMatrix}
    H::TH
    G::TG

    function ClassicalDecodingProblem(H::TH, G::TG) where {TH<:AbstractMatrix,TG<:AbstractMatrix}
        nchecks, nvariables = size(H)
        nchecks == nvariables || throw(DimensionMismatch("classical H must be square"))
        size(G) == (nvariables, nchecks) ||
            throw(DimensionMismatch("classical G must have size ($(nvariables), $(nchecks))"))
        all(isfinite, H) || throw(ArgumentError("H entries must be finite"))
        all(isfinite, G) || throw(ArgumentError("G entries must be finite"))
        return new{TH,TG}(H, G)
    end
end

ClassicalDecodingProblem(H::AbstractMatrix) =
    ClassicalDecodingProblem(H, inv(Matrix{Float64}(H)))

"""
    QuantumDecodingProblem(H, G, logical_check; decision_H=H)

A GKP lattice decoding problem. `H` defines the BP graph, `decision_H` maps a
soft estimate to integer decisions, `G` maps those decisions back to the
received-vector space, and `logical_check` classifies the residual displacement.
Supplying a separate square `decision_H` supports overcomplete BP graphs.
"""
struct QuantumDecodingProblem{
    TH<:AbstractMatrix,
    TG<:AbstractMatrix,
    TL<:AbstractMatrix,
    TD<:AbstractMatrix,
}
    H::TH
    decision_H::TD
    G::TG
    logical_check::TL

    function QuantumDecodingProblem(
        H::TH,
        G::TG,
        logical_check::TL,
        decision_H::TD,
    ) where {TH<:AbstractMatrix,TG<:AbstractMatrix,TL<:AbstractMatrix,TD<:AbstractMatrix}
        _, nvariables = size(H)
        ndecisions, decision_variables = size(decision_H)
        decision_variables == nvariables ||
            throw(DimensionMismatch("decision_H must have one column per variable"))
        size(G) == (nvariables, ndecisions) ||
            throw(DimensionMismatch("G must have size ($(nvariables), $(ndecisions))"))
        size(logical_check, 1) == nvariables ||
            throw(DimensionMismatch("logical_check must have one row per variable"))
        all(isfinite, H) || throw(ArgumentError("H entries must be finite"))
        all(isfinite, decision_H) || throw(ArgumentError("decision_H entries must be finite"))
        all(isfinite, G) || throw(ArgumentError("G entries must be finite"))
        all(isfinite, logical_check) ||
            throw(ArgumentError("logical_check entries must be finite"))
        return new{TH,TG,TL,TD}(H, decision_H, G, logical_check)
    end
end

function QuantumDecodingProblem(
    H::AbstractMatrix,
    G::AbstractMatrix,
    logical_check::AbstractMatrix;
    decision_H::AbstractMatrix=H,
)
    return QuantumDecodingProblem(H, G, logical_check, decision_H)
end

"""
    BinomialEstimate(events, trials, samples; z=1.96)

An event-rate estimate retaining the raw Monte Carlo counts and a clipped
Agresti–Coull confidence interval. `samples` records the number of independently
decoded vectors; for classical symbol-error estimates, `trials` is
`samples * number_of_symbols`.
"""
struct BinomialEstimate
    events::Int
    trials::Int
    samples::Int
    rate::Float64
    lower::Float64
    upper::Float64
    z::Float64
end

function BinomialEstimate(
    events::Integer,
    trials::Integer,
    samples::Integer;
    z::Real=1.96,
)
    trials > 0 || throw(ArgumentError("trials must be positive"))
    samples > 0 || throw(ArgumentError("samples must be positive"))
    0 <= events <= trials || throw(ArgumentError("events must lie between zero and trials"))
    isfinite(z) && z > 0 || throw(ArgumentError("z must be finite and positive"))

    event_count = Int(events)
    trial_count = Int(trials)
    sample_count = Int(samples)
    z_value = Float64(z)
    rate = event_count / trial_count
    adjusted_trials = trial_count + z_value^2
    adjusted_rate = (event_count + z_value^2 / 2) / adjusted_trials
    half_width = z_value * sqrt(adjusted_rate * (1 - adjusted_rate) / adjusted_trials)
    return BinomialEstimate(
        event_count,
        trial_count,
        sample_count,
        rate,
        max(0.0, adjusted_rate - half_width),
        min(1.0, adjusted_rate + half_width),
        z_value,
    )
end

function _validate_simulation(decoder::LDLCDecoder, H::AbstractMatrix, samples::Integer)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    size(H, 2) == decoder.tg.nv ||
        throw(DimensionMismatch("decoder variable count does not match the problem"))
    isfinite(decoder.sigma) && decoder.sigma > 0 ||
        throw(ArgumentError("decoder sigma must be finite and positive"))
    return Int(samples)
end

"""
    estimate_symbol_error_rate!(rng, decoder, problem; samples, z=1.96)

Decode `samples` random classical lattice words with Gaussian noise drawn at
`decoder.sigma`. The mutable decoder is reused serially, and an explicit `rng`
makes the run reproducible.
"""
function estimate_symbol_error_rate!(
    rng::AbstractRNG,
    decoder::LDLCDecoder,
    problem::ClassicalDecodingProblem;
    samples::Integer,
    z::Real=1.96,
)
    sample_count = _validate_simulation(decoder, problem.H, samples)
    n = size(problem.H, 2)
    events = 0

    for _ in 1:sample_count
        symbols = random_bitstring(rng, n)
        received = Vector{Float64}(problem.G * symbols)
        received .+= sample_error(rng, decoder.sigma, n)
        decoded = hard_decision(run_decoder!(decoder, received), problem.H)
        events += count_symbol_errors(decoded, symbols)
    end

    return BinomialEstimate(events, sample_count * n, sample_count; z)
end

function _syndrome_representative(
    error_vector::Vector{Float64},
    problem::QuantumDecodingProblem,
    representative::Symbol,
)
    if representative === :received
        return copy(error_vector)
    elseif representative === :syndrome
        syndrome = mod.(problem.decision_H * error_vector, 1.0)
        return Vector{Float64}(problem.G * syndrome)
    end
    throw(ArgumentError("representative must be :received or :syndrome"))
end

"""
    estimate_logical_error_rate!(rng, decoder, problem; samples,
                                 local_search=nothing,
                                 representative=:received,
                                 atol=1e-5, z=1.96)

Decode `samples` Gaussian displacements and estimate the logical failure rate.
The optional `LocalSearch` is applied after the BP hard decision.
"""
function estimate_logical_error_rate!(
    rng::AbstractRNG,
    decoder::LDLCDecoder,
    problem::QuantumDecodingProblem;
    samples::Integer,
    local_search::Union{Nothing,LocalSearch}=nothing,
    representative::Symbol=:received,
    atol::Real=1e-5,
    z::Real=1.96,
)
    sample_count = _validate_simulation(decoder, problem.H, samples)
    representative in (:received, :syndrome) ||
        throw(ArgumentError("representative must be :received or :syndrome"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    n = size(problem.H, 2)
    failures = 0

    if local_search !== nothing
        size(local_search.G) == size(problem.G) ||
            throw(DimensionMismatch("local-search generator does not match the problem"))
    end

    for _ in 1:sample_count
        error_vector = sample_error(rng, decoder.sigma, n)
        received = _syndrome_representative(error_vector, problem, representative)
        bp_estimate = run_decoder!(decoder, received)
        decision = hard_decision(bp_estimate, problem.decision_H)

        if local_search !== nothing
            reliability = mod.(abs.(problem.decision_H * bp_estimate), 1.0)
            local_search!(received, reliability, decision, local_search)
        end

        correction = received - problem.G * decision
        residual = error_vector - correction
        failures += is_logical_error(problem.logical_check, residual; atol)
    end

    return BinomialEstimate(failures, sample_count, sample_count; z)
end
