using FFTW
using LinearAlgebra
using SparseArrays

"""
    QuantizedMessage(pdf, Δ, origin)

A sampled density on the grid `origin .+ (0:length(pdf)-1) .* Δ`.
"""
mutable struct QuantizedMessage
    pdf::Vector{Float64}
    Δ::Float64
    origin::Float64
end

QuantizedMessage(pdf::Vector{Float64}, Δ::Real) =
    QuantizedMessage(pdf, Float64(Δ), -length(pdf) * Float64(Δ) / 2)

mutable struct CheckNodeQuant
    neighbours::Vector{Tuple{Int64,Float64}}
    messages::Vector{QuantizedMessage} # variable-to-check messages
    pos_in_var_neighbour::Vector{Int64}
end

mutable struct VariableNodeQuant
    id::Int64
    neighbours::Vector{Tuple{Int64,Float64}}
    message::QuantizedMessage          # channel density
    messages::Vector{QuantizedMessage} # check-to-variable messages
    pos_in_check_neighbour::Vector{Int64}
end

"""
    TannerGraphQuant

Mutable state for the discretized-PDF LDLC decoder of Sommer, Feder, and
Shalvi. A graph instance can be reused sequentially, but is not thread-safe.
"""
mutable struct QuantizedDecoderWorkspace{PF,PI}
    folded::Vector{Vector{Float64}}
    spectra::Vector{Vector{ComplexF64}}
    spectral_prefix::Vector{Vector{ComplexF64}}
    spectral_suffix::Vector{Vector{ComplexF64}}
    extrinsic_spectrum::Vector{ComplexF64}
    circle_density::Vector{Float64}
    widened::Vector{Vector{Float64}}
    density_prefix::Vector{Vector{Float64}}
    density_suffix::Vector{Vector{Float64}}
    density_scratch::Vector{Float64}
    posterior::Vector{Float64}
    serial_order::Vector{Int64}
    serial_seen::BitVector
    rfft_plan::PF
    irfft_plan::PI
end

function QuantizedDecoderWorkspace(
    L::Int,
    period_bins::Int,
    max_check_degree::Int,
    max_variable_degree::Int,
    variable_count::Int,
)
    check_degree = max(1, max_check_degree)
    variable_degree = max(1, max_variable_degree)
    spectrum_length = period_bins ÷ 2 + 1
    folded = [zeros(Float64, period_bins) for _ in 1:check_degree]
    spectra = [zeros(ComplexF64, spectrum_length) for _ in 1:check_degree]
    spectral_prefix = [zeros(ComplexF64, spectrum_length) for _ in 1:(check_degree + 1)]
    spectral_suffix = [zeros(ComplexF64, spectrum_length) for _ in 1:(check_degree + 1)]
    extrinsic_spectrum = zeros(ComplexF64, spectrum_length)
    circle_density = zeros(Float64, period_bins)
    widened = [zeros(Float64, L) for _ in 1:variable_degree]
    density_prefix = [zeros(Float64, L) for _ in 1:(variable_degree + 1)]
    density_suffix = [zeros(Float64, L) for _ in 1:(variable_degree + 1)]
    density_scratch = zeros(Float64, L)
    posterior = zeros(Float64, L)
    serial_order = Vector{Int64}(undef, variable_count)
    serial_seen = falses(variable_count)

    rfft_input = zeros(Float64, period_bins)
    rfft_output = zeros(ComplexF64, spectrum_length)
    rfft_plan = plan_rfft(rfft_input; flags=FFTW.ESTIMATE)
    irfft_plan = plan_irfft(rfft_output, period_bins; flags=FFTW.ESTIMATE)

    return QuantizedDecoderWorkspace(
        folded,
        spectra,
        spectral_prefix,
        spectral_suffix,
        extrinsic_spectrum,
        circle_density,
        widened,
        density_prefix,
        density_suffix,
        density_scratch,
        posterior,
        serial_order,
        serial_seen,
        rfft_plan,
        irfft_plan,
    )
end

mutable struct TannerGraphQuant{W}
    var_nodes::Vector{VariableNodeQuant}
    check_nodes::Vector{CheckNodeQuant}
    var_node_to_posit::Dict{Int64,Int64}
    nv::Int64
    nc::Int64
    L::Int
    Δ::Float64
    grid::Vector{Float64} # relative grid; message grids carry absolute origins
    bp_result::Vector{Float64}
    period_bins::Int
    widen::Bool
    schedule::Vector{Int64}
    workspace::W
end

make_grid(Δ::Float64, L::Int) = collect((-L ÷ 2):(L ÷ 2 - 1)) .* Δ

function _validate_quantization(L::Int, Δ::Float64)
    L > 0 || throw(ArgumentError("L must be positive"))
    iseven(L) || throw(ArgumentError("L must be even"))
    isfinite(Δ) && Δ > 0 || throw(ArgumentError("Δ must be finite and positive"))
    reciprocal = inv(Δ)
    period_bins = round(Int, reciprocal)
    isapprox(reciprocal, period_bins; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("1/Δ must be an integer for the periodic FFT update"))
    period_bins >= 2 || throw(ArgumentError("1/Δ must be at least two"))
    return period_bins
end

function _normalize_pdf!(pdf::Vector{Float64}, Δ::Float64)
    tolerance = 100eps(Float64) * max(1.0, maximum(abs, pdf; init=0.0))
    @inbounds for index in eachindex(pdf)
        value = pdf[index]
        isfinite(value) || throw(ArgumentError("quantized message contains a non-finite density"))
        value >= -tolerance || throw(ArgumentError("quantized message contains a negative density"))
        value < 0 && (pdf[index] = 0.0)
    end
    mass = sum(pdf) * Δ
    isfinite(mass) && mass > 0 ||
        throw(ArgumentError("quantized message has zero or non-finite total mass"))
    pdf ./= mass
    return pdf
end

normalize!(message::QuantizedMessage) =
    (_normalize_pdf!(message.pdf, message.Δ); nothing)

_grid_value(message::QuantizedMessage, index::Int) =
    message.origin + (index - 1) * message.Δ

function _empty_message(L::Int, Δ::Float64)
    return QuantizedMessage(zeros(Float64, L), Δ, -L * Δ / 2)
end

"""
    initialize_tanner_graph_quant(H; L=256, Δ=1/64, widen=true)

Construct the graph for the discretized-PDF LDLC decoder. Unlike the original
regular-LDLC presentation, rectangular graphs, isolated variables, and
degree-one nodes are supported.
"""
function initialize_tanner_graph_quant(
    H::SparseMatrixCSC{<:Real};
    L::Int=256,
    Δ::Real=1 / 64,
    widen::Bool=true,
)
    step = Float64(Δ)
    period_bins = _validate_quantization(L, step)
    nc, nv = size(H)

    variable_neighbours = [Tuple{Int64,Float64}[] for _ in 1:nv]
    variable_positions = [Int64[] for _ in 1:nv]
    check_nodes = Vector{CheckNodeQuant}(undef, nc)

    for check_index in 1:nc
        row = H[check_index, :]
        neighbours = Tuple{Int64,Float64}[]
        positions = Int64[]
        for row_position in eachindex(row.nzind)
            variable_index = Int64(row.nzind[row_position])
            weight = Float64(row.nzval[row_position])
            isfinite(weight) && !iszero(weight) ||
                throw(ArgumentError("H[$check_index,$variable_index] must be finite and nonzero"))
            push!(neighbours, (variable_index, weight))
            push!(variable_neighbours[variable_index], (Int64(check_index), weight))
            push!(variable_positions[variable_index], Int64(row_position))
            push!(positions, Int64(length(variable_neighbours[variable_index])))
        end
        check_nodes[check_index] = CheckNodeQuant(
            neighbours,
            [_empty_message(L, step) for _ in neighbours],
            positions,
        )
    end

    variable_nodes = Vector{VariableNodeQuant}(undef, nv)
    for variable_index in 1:nv
        neighbours = variable_neighbours[variable_index]
        variable_nodes[variable_index] = VariableNodeQuant(
            Int64(variable_index),
            neighbours,
            _empty_message(L, step),
            [_empty_message(L, step) for _ in neighbours],
            variable_positions[variable_index],
        )
    end

    max_check_degree = maximum(length(check.neighbours) for check in check_nodes; init=0)
    max_variable_degree = maximum(length(variable.neighbours) for variable in variable_nodes; init=0)
    workspace = QuantizedDecoderWorkspace(L, period_bins, max_check_degree, max_variable_degree, nv)

    return TannerGraphQuant(
        variable_nodes,
        check_nodes,
        Dict{Int64,Int64}(Int64(index) => Int64(index) for index in 1:nv),
        Int64(nv),
        Int64(nc),
        L,
        step,
        make_grid(step, L),
        zeros(Float64, nv),
        period_bins,
        widen,
        collect(Int64, 1:nv),
        workspace,
    )
end

initialize_tanner_graph_quant(
    H::AbstractMatrix{<:Real};
    L::Int=256,
    Δ::Real=1 / 64,
    widen::Bool=true,
) = initialize_tanner_graph_quant(sparse(H); L=L, Δ=Δ, widen=widen)

# Preserve the positional dense-matrix constructor used by older examples.
initialize_tanner_graph_quant(H::AbstractMatrix{<:Real}, L::Int, Δ::Real) =
    initialize_tanner_graph_quant(H; L=L, Δ=Δ)

function _validate_observations(graph::TannerGraphQuant, observations, sigma)
    length(observations) == graph.nv || throw(DimensionMismatch(
        "received vector has length $(length(observations)); expected $(graph.nv)",
    ))
    length(sigma) == graph.nv || throw(DimensionMismatch(
        "noise vector has length $(length(sigma)); expected $(graph.nv)",
    ))
    all(isfinite, observations) || throw(ArgumentError("received values must be finite"))
    all(value -> isfinite(value) && value > 0, sigma) ||
        throw(ArgumentError("noise standard deviations must be finite and positive"))
end

function _initialize_messages!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma::AbstractVector{<:Real},
)
    _validate_observations(graph, observations, sigma)
    return _initialize_messages_unchecked!(graph, observations, sigma)
end

function _initialize_messages_unchecked!(graph::TannerGraphQuant, observations, sigma)
    for variable in graph.var_nodes
        mean = Float64(observations[variable.id])
        standard_deviation = sigma isa Real ? Float64(sigma) : Float64(sigma[variable.id])
        origin = mean - graph.L * graph.Δ / 2
        channel = variable.message
        channel.origin = origin
        @inbounds for index in eachindex(channel.pdf)
            coordinate = _grid_value(channel, index)
            channel.pdf[index] = exp(-0.5 * ((coordinate - mean) / standard_deviation)^2)
        end
        _normalize_pdf!(channel.pdf, graph.Δ)

        for incoming in variable.messages
            incoming.origin = origin
            fill!(incoming.pdf, inv(graph.L * graph.Δ))
        end
    end

    for (check_index, check) in enumerate(graph.check_nodes)
        for edge_index in eachindex(check.neighbours)
            variable_index, _ = check.neighbours[edge_index]
            channel = graph.var_nodes[variable_index].message
            outgoing = check.messages[edge_index]
            outgoing.origin = channel.origin
            copyto!(outgoing.pdf, channel.pdf)
        end
    end
    return nothing
end

function initialize_messages!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma::Real,
)
    isfinite(sigma) && sigma > 0 ||
        throw(ArgumentError("noise standard deviation must be finite and positive"))
    length(observations) == graph.nv || throw(DimensionMismatch(
        "received vector has length $(length(observations)); expected $(graph.nv)",
    ))
    all(isfinite, observations) || throw(ArgumentError("received values must be finite"))
    return _initialize_messages_unchecked!(graph, observations, sigma)
end

initialize_messages!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma::AbstractVector{<:Real},
) = _initialize_messages!(graph, observations, sigma)

"""Push a sampled density through `x -> h*x mod 1`."""
function _fold_message!(
    folded::Vector{Float64},
    message::QuantizedMessage,
    weight::Float64,
)
    period_bins = length(folded)
    isapprox(message.Δ * period_bins, 1.0; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("folded density and message have incompatible resolutions"))
    fill!(folded, 0.0)
    @inbounds for index in eachindex(message.pdf)
        coordinate = _grid_value(message, index)
        bin_coordinate = mod(weight * coordinate, 1.0) * period_bins
        lower_zero_based = floor(Int, bin_coordinate)
        fraction = bin_coordinate - lower_zero_based
        lower = mod(lower_zero_based, period_bins) + 1
        upper = mod(lower_zero_based + 1, period_bins) + 1
        density = message.pdf[index]
        folded[lower] += (1 - fraction) * density
        folded[upper] += fraction * density
    end
    _normalize_pdf!(folded, message.Δ)
    return folded
end

function _periodic_sample(density::Vector{Float64}, coordinate::Float64)
    bins = length(density)
    bin_coordinate = mod(coordinate, 1.0) * bins
    lower_zero_based = floor(Int, bin_coordinate)
    fraction = bin_coordinate - lower_zero_based
    lower = mod(lower_zero_based, bins) + 1
    upper = mod(lower_zero_based + 1, bins) + 1
    return (1 - fraction) * density[lower] + fraction * density[upper]
end

function _unit_mass_spectrum!(spectrum::Vector{ComplexF64}, density::Vector{Float64}, graph::TannerGraphQuant)
    LinearAlgebra.mul!(spectrum, graph.workspace.rfft_plan, density)
    # A normalized sampled density has sum(density) == 1/Δ. Scaling its
    # spectrum by Δ therefore makes the DC component one and prevents the
    # check-node product from growing as (1/Δ)^degree.
    @inbounds for index in eachindex(spectrum)
        spectrum[index] *= graph.Δ
    end
    return spectrum
end

function _write_periodic_check_message!(
    output::QuantizedMessage,
    circle_density::Vector{Float64},
    weight::Float64,
)
    @inbounds for index in eachindex(output.pdf)
        coordinate = _grid_value(output, index)
        output.pdf[index] = _periodic_sample(circle_density, -weight * coordinate)
    end
    _normalize_pdf!(output.pdf, output.Δ)
    return output
end

function _check_output(graph::TannerGraphQuant, check::CheckNodeQuant, edge_index::Int)
    variable_index, weight = check.neighbours[edge_index]
    variable_position = check.pos_in_var_neighbour[edge_index]
    output = graph.var_nodes[variable_index].messages[variable_position]
    return output, weight
end

function _write_degree_one_check_message!(graph::TannerGraphQuant, check::CheckNodeQuant, edge_index::Int)
    circle_density = graph.workspace.circle_density
    fill!(circle_density, 0.0)
    circle_density[1] = inv(graph.Δ)
    output, weight = _check_output(graph, check, edge_index)
    _write_periodic_check_message!(output, circle_density, weight)
    return nothing
end

"""
Compute every check-to-variable message for one check using the Appendix VIII
unit-period FFT formulation. Prefix and suffix spectral products avoid FFT
division and remain valid when a frequency bin is zero.
"""
function check_node_messages!(graph::TannerGraphQuant, check_index::Int64)
    check = graph.check_nodes[check_index]
    degree = length(check.neighbours)
    iszero(degree) && return nothing
    degree == 1 && return _write_degree_one_check_message!(graph, check, 1)

    workspace = graph.workspace
    folded = workspace.folded

    if degree == 2
        for edge_index in 1:2
            incoming_edge = 3 - edge_index
            _, incoming_weight = check.neighbours[incoming_edge]
            _fold_message!(folded[1], check.messages[incoming_edge], incoming_weight)
            output, weight = _check_output(graph, check, edge_index)
            _write_periodic_check_message!(output, folded[1], weight)
        end
        return nothing
    end

    spectra = workspace.spectra
    for edge_index in 1:degree
        _, weight = check.neighbours[edge_index]
        _fold_message!(folded[edge_index], check.messages[edge_index], weight)
        _unit_mass_spectrum!(spectra[edge_index], folded[edge_index], graph)
    end

    spectrum_length = length(spectra[1])
    prefix = workspace.spectral_prefix
    suffix = workspace.spectral_suffix
    fill!(prefix[1], one(ComplexF64))
    for edge_index in 1:degree
        @inbounds for frequency in 1:spectrum_length
            prefix[edge_index + 1][frequency] = prefix[edge_index][frequency] * spectra[edge_index][frequency]
        end
    end
    fill!(suffix[degree + 1], one(ComplexF64))
    for edge_index in degree:-1:1
        @inbounds for frequency in 1:spectrum_length
            suffix[edge_index][frequency] = spectra[edge_index][frequency] * suffix[edge_index + 1][frequency]
        end
    end

    extrinsic_spectrum = workspace.extrinsic_spectrum
    circle_density = workspace.circle_density
    for edge_index in 1:degree
        @inbounds for frequency in 1:spectrum_length
            extrinsic_spectrum[frequency] = prefix[edge_index][frequency] * suffix[edge_index + 1][frequency]
        end
        LinearAlgebra.mul!(circle_density, workspace.irfft_plan, extrinsic_spectrum)
        _normalize_pdf!(circle_density, graph.Δ)
        output, weight = _check_output(graph, check, edge_index)
        _write_periodic_check_message!(output, circle_density, weight)
    end
    return nothing
end

"""
Compute one check-to-variable message from the current variable-to-check
messages. This directed form is used by the asynchronous serial schedule.
"""
function check_node_message!(
    graph::TannerGraphQuant,
    check_index::Int64,
    edge_index::Int64,
)
    check = graph.check_nodes[check_index]
    degree = length(check.neighbours)
    1 <= edge_index <= degree || throw(BoundsError(check.neighbours, edge_index))
    degree == 1 && return _write_degree_one_check_message!(graph, check, edge_index)

    workspace = graph.workspace
    folded = workspace.folded[1]

    if degree == 2
        incoming_edge = 3 - edge_index
        _, incoming_weight = check.neighbours[incoming_edge]
        _fold_message!(folded, check.messages[incoming_edge], incoming_weight)
        output, weight = _check_output(graph, check, edge_index)
        _write_periodic_check_message!(output, folded, weight)
        return nothing
    end

    spectrum = workspace.extrinsic_spectrum
    temporary_spectrum = workspace.spectra[1]
    fill!(spectrum, one(ComplexF64))
    for incoming_edge in 1:degree
        incoming_edge == edge_index && continue
        _, incoming_weight = check.neighbours[incoming_edge]
        _fold_message!(folded, check.messages[incoming_edge], incoming_weight)
        _unit_mass_spectrum!(temporary_spectrum, folded, graph)
        @inbounds for frequency in eachindex(spectrum)
            spectrum[frequency] *= temporary_spectrum[frequency]
        end
    end

    circle_density = workspace.circle_density
    LinearAlgebra.mul!(circle_density, workspace.irfft_plan, spectrum)
    _normalize_pdf!(circle_density, graph.Δ)
    output, weight = _check_output(graph, check, edge_index)
    _write_periodic_check_message!(output, circle_density, weight)
    return nothing
end

function check_node_iterations!(graph::TannerGraphQuant)
    for check_index in 1:graph.nc
        check_node_messages!(graph, check_index)
    end
    return nothing
end

function _widen_density!(output::Vector{Float64}, input::Vector{Float64})
    length(output) == length(input) || throw(DimensionMismatch("widening buffers differ in length"))
    output !== input || throw(ArgumentError("widening output must not alias its input"))
    last_index = lastindex(input)
    @inbounds @simd for index in eachindex(input)
        left = index == 1 ? 0.0 : input[index - 1]
        right = index == last_index ? 0.0 : input[index + 1]
        output[index] = left + input[index] + right
    end
    return output
end

function _multiply_density_to!(
    output::Vector{Float64},
    left::Vector{Float64},
    right::Vector{Float64},
    graph::TannerGraphQuant,
)
    mass_sum = 0.0
    minimum_input = Inf
    @inbounds @simd for index in eachindex(output, left, right)
        left_value = left[index]
        right_value = right[index]
        minimum_input = min(minimum_input, min(left_value, right_value))
        value = left_value * right_value
        output[index] = value
        mass_sum += value
    end
    minimum_input >= 0 ||
        throw(ArgumentError("quantized message contains a negative or non-finite density"))
    mass = mass_sum * graph.Δ
    isfinite(mass) && mass > 0 ||
        throw(ArgumentError("quantized message has zero or non-finite total mass"))
    inverse_mass = inv(mass)
    @inbounds @simd for index in eachindex(output)
        output[index] *= inverse_mass
    end
    return output
end

_multiply_density!(product::Vector{Float64}, incoming::Vector{Float64}, graph::TannerGraphQuant) =
    _multiply_density_to!(product, product, incoming, graph)

function _prepare_variable_densities!(graph::TannerGraphQuant, variable::VariableNodeQuant)
    widened = graph.workspace.widened
    for edge_index in eachindex(variable.messages)
        if graph.widen
            _widen_density!(widened[edge_index], variable.messages[edge_index].pdf)
        else
            copyto!(widened[edge_index], variable.messages[edge_index].pdf)
        end
    end
    return widened
end

function _variable_output(graph::TannerGraphQuant, variable::VariableNodeQuant, edge_index::Int)
    check_index, _ = variable.neighbours[edge_index]
    check_position = variable.pos_in_check_neighbour[edge_index]
    return graph.check_nodes[check_index].messages[check_position]
end

function variable_node_messages!(graph::TannerGraphQuant, variable_index::Int64)
    variable = graph.var_nodes[variable_index]
    degree = length(variable.neighbours)
    iszero(degree) && return nothing

    if degree == 1
        output = _variable_output(graph, variable, 1)
        output.origin = variable.message.origin
        copyto!(output.pdf, variable.message.pdf)
        return nothing
    end

    workspace = graph.workspace
    widened = _prepare_variable_densities!(graph, variable)

    if degree == 2
        for excluded_edge in 1:2
            incoming_edge = 3 - excluded_edge
            output = _variable_output(graph, variable, excluded_edge)
            _multiply_density_to!(output.pdf, variable.message.pdf, widened[incoming_edge], graph)
            output.origin = variable.message.origin
        end
        return nothing
    end

    prefix = workspace.density_prefix
    suffix = workspace.density_suffix
    copyto!(prefix[1], variable.message.pdf)

    for edge_index in 1:(degree - 1)
        _multiply_density_to!(
            prefix[edge_index + 1],
            prefix[edge_index],
            widened[edge_index],
            graph,
        )
    end

    fill!(suffix[degree + 1], 1.0)
    for edge_index in degree:-1:2
        _multiply_density_to!(
            suffix[edge_index],
            suffix[edge_index + 1],
            widened[edge_index],
            graph,
        )
    end

    for excluded_edge in 1:degree
        output = _variable_output(graph, variable, excluded_edge)
        _multiply_density_to!(
            output.pdf,
            prefix[excluded_edge],
            suffix[excluded_edge + 1],
            graph,
        )
        output.origin = variable.message.origin
    end
    return nothing
end

function variable_node_iterations!(graph::TannerGraphQuant)
    for variable_index in 1:graph.nv
        variable_node_messages!(graph, variable_index)
    end
    return nothing
end

function _serial_variable_node_update!(graph::TannerGraphQuant, variable_index::Int64)
    variable = graph.var_nodes[variable_index]
    for variable_edge in eachindex(variable.neighbours)
        check_index, _ = variable.neighbours[variable_edge]
        check_edge = variable.pos_in_check_neighbour[variable_edge]
        check_node_message!(graph, check_index, check_edge)
    end
    variable_node_messages!(graph, variable_index)
    return nothing
end

function _validate_serial_schedule(graph::TannerGraphQuant, schedule)
    length(schedule) == graph.nv || throw(ArgumentError(
        "serial schedule must contain each of the $(graph.nv) variables exactly once",
    ))
    order = graph.workspace.serial_order
    seen = graph.workspace.serial_seen
    fill!(seen, false)
    for (position, raw_index) in enumerate(schedule)
        raw_index isa Integer || throw(ArgumentError(
            "serial schedule entry $position is not an integer variable index",
        ))
        variable_index = Int64(raw_index)
        1 <= variable_index <= graph.nv || throw(ArgumentError(
            "serial schedule entry $position is $variable_index; expected an index in 1:$(graph.nv)",
        ))
        !seen[variable_index] || throw(ArgumentError(
            "serial schedule contains variable $variable_index more than once",
        ))
        order[position] = variable_index
        seen[variable_index] = true
    end
    return order
end

function variable_node_decision!(graph::TannerGraphQuant, variable_index::Int64)
    variable = graph.var_nodes[variable_index]
    posterior = graph.workspace.posterior
    scratch = graph.workspace.density_scratch
    copyto!(posterior, variable.message.pdf)
    for incoming in variable.messages
        if graph.widen
            _widen_density!(scratch, incoming.pdf)
            _multiply_density!(posterior, scratch, graph)
        else
            _multiply_density!(posterior, incoming.pdf, graph)
        end
    end
    maximum_index = argmax(posterior)
    graph.bp_result[variable.id] = _grid_value(variable.message, maximum_index)
    return nothing
end

function decision_step!(graph::TannerGraphQuant)
    for variable_index in 1:graph.nv
        variable_node_decision!(graph, variable_index)
    end
    return nothing
end

function _run_sommer_decoder!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma,
    max_iter::Integer,
)
    max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative"))
    initialize_messages!(graph, observations, sigma)
    for _ in 1:max_iter
        check_node_iterations!(graph)
        variable_node_iterations!(graph)
    end
    decision_step!(graph)
    return graph.bp_result
end

"""
    run_belief_propagation!(graph::TannerGraphQuant, observations, sigma, max_iter)

Run the parallel discretized-PDF decoder from Sommer, Feder, and Shalvi,
*Low-Density Lattice Codes*, IEEE Transactions on Information Theory 54(4),
2008. The returned vector contains the peak location of each final marginal;
use [`hard_decision`](@ref) to recover the integer vector.
"""
run_belief_propagation!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma::Real,
    max_iter::Integer,
) = _run_sommer_decoder!(graph, observations, sigma, max_iter)

function _run_sommer_serial_decoder!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma,
    max_iter::Integer,
    schedule,
)
    max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative"))
    order = _validate_serial_schedule(graph, schedule)
    initialize_messages!(graph, observations, sigma)
    for _ in 1:max_iter
        for variable_index in order
            _serial_variable_node_update!(graph, variable_index)
        end
    end
    decision_step!(graph)
    return graph.bp_result
end

"""
    run_serial_belief_propagation!(
        graph::TannerGraphQuant, observations, sigma, max_iter;
        schedule=graph.schedule,
    )

Run asynchronous (Gauss–Seidel) discretized-PDF belief propagation. During
each sweep, a variable first receives freshly computed messages from all of
its adjacent checks and then immediately updates all of its outgoing messages.
`schedule` must be a permutation of the variable indices and is reused for
every sweep.
"""
function run_serial_belief_propagation!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma::Real,
    max_iter::Integer;
    schedule=graph.schedule,
)
    return _run_sommer_serial_decoder!(graph, observations, sigma, max_iter, schedule)
end

function run_serial_belief_propagation!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma::AbstractVector{<:Real},
    max_iter::Integer;
    schedule=graph.schedule,
)
    return _run_sommer_serial_decoder!(graph, observations, sigma, max_iter, schedule)
end

run_belief_propagation!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma::AbstractVector{<:Real},
    max_iter::Integer,
) = _run_sommer_decoder!(graph, observations, sigma, max_iter)

"""
    _run_sommer_horizontal_decoder!(graph, observations, sigma, max_iter)

Run a check-layered quantized schedule. Each check first refreshes all of its
outgoing messages, after which its neighbouring variables immediately refresh
their variable-to-check messages. This mirrors `LDLCDecoder`'s
`:serial_horizontal` schedule for Gaussian messages.
"""
function _run_sommer_horizontal_decoder!(
    graph::TannerGraphQuant,
    observations::AbstractVector{<:Real},
    sigma,
    max_iter::Integer,
)
    max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative"))
    initialize_messages!(graph, observations, sigma)
    for _ in 1:max_iter
        for check_index in 1:graph.nc
            check_node_messages!(graph, Int64(check_index))
            for (variable_index, _) in graph.check_nodes[check_index].neighbours
                variable_node_messages!(graph, variable_index)
            end
        end
    end
    decision_step!(graph)
    return graph.bp_result
end
