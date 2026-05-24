mutable struct FBAlloc
    forward::Vector{Vector{gaussian_log_weight}}
    backward::Vector{Vector{gaussian_log_weight}}
    mixtures::Vector{Vector{gaussian_log_weight}}
    combined::Vector{gaussian_log_weight}
    outputs::Vector{gaussian_log_weight}
    ws_alloc::Vector{Float64}
end

function FBAlloc(d::Int, M::Int)
    mixtures = [map(copy, fill(gaussian_log_weight(0.0, 1.0), M)) for _ in 1:d]
    forward = [map(copy, fill(gaussian_log_weight(0.0, 1.0), M^k)) for k = 1:d]
    backward = [map(copy, fill(gaussian_log_weight(0.0, 1.0), M^k)) for k = d:-1:1]
    combined = map(copy, fill(gaussian_log_weight(0.0, 1.0), M^(d - 1)))
    outputs = map(copy, fill(gaussian_log_weight(0.0, 1.0), d))
    ws_alloc = Vector{Float64}(undef, M^d)
    return FBAlloc(forward, backward, mixtures, combined, outputs, ws_alloc)
end

mutable struct LDLCDecoder
    tg::TannerGraph
    schedule::Symbol
    algorithm::Union{Symbol,Int64}
    sigma::Float64
    max_iterations::Int64
    search_interval::Float64
    lsd_beta::Float64
    lsd_w_min::Float64
    m_gaussian_allocs::Dict{Int,FBAlloc}
end

function LDLCDecoder(
    tg::TannerGraph;
    schedule=:parallel,
    algorithm=:lsd,
    sigma::Float64=1.0,
    max_iterations::Int64=25,
    search_interval::Float64=1.5,
    lsd_beta::Float64=LatticeDecoder.LSD_DEFAULT_BETA,
    lsd_w_min::Float64=LatticeDecoder.LSD_W_MIN,
    M=nothing,
)
    decoder_schedule = _normalize_decoder_schedule(schedule)
    decoder_algorithm = _normalize_decoder_algorithm(algorithm, M)
    _validate_decoder_config(decoder_schedule, decoder_algorithm)
    _validate_lsd_beta(lsd_beta)
    _validate_lsd_w_min(lsd_w_min)
    allocs = decoder_algorithm isa Int64 ? _m_gaussian_allocs(tg, decoder_algorithm) : Dict{Int,FBAlloc}()
    return LDLCDecoder(tg, decoder_schedule, decoder_algorithm, sigma, max_iterations, search_interval, lsd_beta, lsd_w_min, allocs)
end

LDLCDecoder(tg::TannerGraph, M::Int) = LDLCDecoder(tg; schedule=:parallel, algorithm=Int64(M))

function _normalize_decoder_schedule(schedule::Symbol)
    return schedule
end

function _normalize_decoder_schedule(schedule::AbstractString)
    return Symbol(schedule)
end

function _normalize_decoder_schedule(schedule)
    return schedule
end

function _normalize_decoder_algorithm(algorithm, M)
    if M !== nothing
        return Int64(M)
    elseif algorithm isa AbstractString
        return Symbol(algorithm)
    elseif algorithm isa Integer
        return Int64(algorithm)
    else
        return algorithm
    end
end

function _validate_decoder_config(schedule::Symbol, algorithm::Union{Symbol,Int64})
    schedule in (:parallel, :serial) || throw(ArgumentError("Unsupported schedule $(schedule). Choose :parallel or :serial."))
    if algorithm isa Symbol
        algorithm in (:nearest, :lsd) || throw(ArgumentError("Unsupported algorithm $(algorithm). Choose :nearest, :lsd, or an integer M."))
    else
        algorithm > 0 || throw(ArgumentError("M-Gaussian algorithm requires a positive integer M."))
    end
    return nothing
end

function _validate_decoder_config(schedule, algorithm)
    throw(ArgumentError("Unsupported decoder configuration. Choose schedule :parallel or :serial and algorithm :nearest, :lsd, or an integer M."))
end

function _m_gaussian_allocs(tg::TannerGraph, M::Int64)
    degrees = unique(length(vn.neighbours) for vn in tg.var_nodes)
    return Dict(d => FBAlloc(d, M) for d in degrees)
end

function _ensure_m_gaussian_allocs!(dec::LDLCDecoder)
    dec.algorithm isa Int64 || return nothing
    if isempty(dec.m_gaussian_allocs)
        dec.m_gaussian_allocs = _m_gaussian_allocs(dec.tg, dec.algorithm)
    end
    return nothing
end

"""
    run_decoder!(dec::LDLCDecoder, message::Vector{Float64})

Run the decoder with the mutable configuration stored on `dec`.
"""
function run_decoder!(dec::LDLCDecoder, message::Vector{Float64})
    _validate_decoder_config(dec.schedule, dec.algorithm)
    _validate_lsd_beta(dec.lsd_beta)
    _validate_lsd_w_min(dec.lsd_w_min)
    _ensure_m_gaussian_allocs!(dec)

    initialize_messages!(dec.tg, message, dec.sigma)
    dec.tg.search_interval = dec.search_interval
    dec.tg.lsd_beta = dec.lsd_beta
    dec.tg.lsd_w_min = dec.lsd_w_min

    if dec.schedule == :parallel
        _run_parallel_decoder!(dec)
    else
        _run_serial_decoder!(dec)
    end

    _decision_step!(dec)
    return dec.tg.bp_result
end

function _run_parallel_decoder!(dec::LDLCDecoder)
    @inbounds for _ = 1:dec.max_iterations
        check_node_iterations!(dec.tg)
        _variable_node_iterations!(dec)
    end
    return nothing
end

function _run_serial_decoder!(dec::LDLCDecoder)
    @inbounds for _ = 1:dec.max_iterations
        update_reliability_schedule!(dec.tg)
        for vn_idx in dec.tg.schedule
            _update_serial_variable_node!(dec, vn_idx)
        end
    end
    return nothing
end

function _variable_node_iterations!(dec::LDLCDecoder)
    @inbounds for vn_idx = 1:dec.tg.nv
        _variable_node_messages!(dec, vn_idx)
    end
    return nothing
end

function _update_serial_variable_node!(dec::LDLCDecoder, vn_idx::Int64)
    vn = dec.tg.var_nodes[vn_idx]

    @inbounds for j = 1:length(vn.neighbours)
        cn_idx, _ = vn.neighbours[j]
        vn_pos_idx = vn.pos_in_check_neighbour[j]
        check_node_message!(vn, dec.tg, cn_idx, j, vn_pos_idx)
    end

    _variable_node_messages!(dec, vn_idx)
    return nothing
end

function _variable_node_messages!(dec::LDLCDecoder, vn_idx::Int64)
    if dec.algorithm == :nearest
        variable_node_messages_allocationless!(dec.tg, vn_idx)
    elseif dec.algorithm == :lsd
        lsd_variable_node_messages!(dec.tg, vn_idx)
    else
        variable_node_messages_M_gaussian_allocationless!(dec.tg, vn_idx, dec)
    end
    return nothing
end

function _decision_step!(dec::LDLCDecoder)
    if dec.algorithm == :lsd
        decision_step_lsd!(dec.tg)
    elseif dec.algorithm isa Int64
        decision_step_M_gaussian_allocationless!(dec)
    else
        decision_step_nearest!(dec.tg)
    end
    return nothing
end

"""
    variable_node_messages_M_gaussian_allocationless!(tg, vn_idx, decoder)

Allocation-conscious M-Gaussian variable node update using workspaces owned by `decoder`.
"""
function variable_node_messages_M_gaussian_allocationless!(tg::TannerGraph, vn_idx::Int64, decoder::LDLCDecoder)
    var_node = tg.var_nodes[vn_idx]
    d = length(var_node.neighbours)

    if d == 1
        cn_idx, _ = var_node.neighbours[1]
        idx = var_node.pos_in_check_neighbour[1]
        cn = tg.check_nodes[cn_idx]

        cn.messages[idx].mean = var_node.message.mean
        cn.messages[idx].var = var_node.message.var
    else
        alloc = decoder.m_gaussian_allocs[d]
        M = decoder.algorithm::Int64

        @inbounds for j = 1:d
            m_nearest!(alloc.mixtures[j], var_node.messages[j], var_node.message.mean, M)
        end

        forward_backward_recursion!(decoder, var_node.message, d)

        @inbounds for j = 1:d
            cn_idx, _ = var_node.neighbours[j]
            idx = var_node.pos_in_check_neighbour[j]
            cn = tg.check_nodes[cn_idx]

            cn.messages[idx].mean = alloc.outputs[j].mean
            cn.messages[idx].var = alloc.outputs[j].var
        end
    end

    return nothing
end

"""
    forward_backward_recursion!(decoder, g, d)

Forward-backward recursion for the M-Gaussian variable node update.
"""
function forward_backward_recursion!(decoder::LDLCDecoder, g::gaussian_log_weight, d::Int64)
    alloc = decoder.m_gaussian_allocs[d]
    forward = alloc.forward
    backward = alloc.backward
    mixtures = alloc.mixtures
    combined = alloc.combined
    outputs = alloc.outputs
    ws_alloc = alloc.ws_alloc

    forward[1] .= mixtures[1]
    @inbounds for j = 2:d
        prod!(forward[j], forward[j - 1], mixtures[j])
    end

    backward[d] .= mixtures[d]
    @inbounds for j = (d - 1):-1:1
        prod!(backward[j], backward[j + 1], mixtures[j])
    end

    @inbounds for j = 1:d
        if j == 1
            prod!(combined, backward[j + 1], g)
        elseif j == d
            prod!(combined, forward[j - 1], g)
        else
            prod!(combined, forward[j - 1], backward[j + 1])
            prod!(combined, g)
        end
        moment_matching!(outputs[j], combined, ws_alloc)
    end

    return nothing
end

function variable_node_decision_M_gaussian_allocationless!(dec::LDLCDecoder, vn_idx::Int64)
    var_node = dec.tg.var_nodes[vn_idx]
    d = length(var_node.neighbours)
    alloc = dec.m_gaussian_allocs[d]
    M = dec.algorithm::Int64

    @inbounds for j = 1:d
        m_nearest!(alloc.mixtures[j], var_node.messages[j], var_node.message.mean, M)
    end

    alloc.forward[1] .= alloc.mixtures[1]
    @inbounds for j = 2:d
        prod!(alloc.forward[j], alloc.forward[j - 1], alloc.mixtures[j])
    end

    prod!(alloc.forward[d], var_node.message)
    moment_matching!(var_node.message, alloc.forward[d], alloc.ws_alloc)
    dec.tg.bp_result[vn_idx] = var_node.message.mean

    return nothing
end

function decision_step_M_gaussian_allocationless!(dec::LDLCDecoder)
    @inbounds for vn_idx = 1:dec.tg.nv
        variable_node_decision_M_gaussian_allocationless!(dec, vn_idx)
    end
    return nothing
end

function run_decoder_parallel!(dec::LDLCDecoder, message::Vector{Float64}, sigma::Float64, max_iter::Int64; lsd_beta::Float64=dec.lsd_beta, lsd_w_min::Float64=dec.lsd_w_min)
    _validate_lsd_beta(lsd_beta)
    _validate_lsd_w_min(lsd_w_min)
    dec.schedule = :parallel
    dec.sigma = sigma
    dec.max_iterations = max_iter
    dec.lsd_beta = lsd_beta
    dec.lsd_w_min = lsd_w_min
    return run_decoder!(dec, message)
end

function run_decoder_serial!(dec::LDLCDecoder, message::Vector{Float64}, sigma::Float64, max_iter::Int64; lsd_beta::Float64=dec.lsd_beta, lsd_w_min::Float64=dec.lsd_w_min)
    _validate_lsd_beta(lsd_beta)
    _validate_lsd_w_min(lsd_w_min)
    dec.schedule = :serial
    dec.sigma = sigma
    dec.max_iterations = max_iter
    dec.lsd_beta = lsd_beta
    dec.lsd_w_min = lsd_w_min
    return run_decoder!(dec, message)
end
