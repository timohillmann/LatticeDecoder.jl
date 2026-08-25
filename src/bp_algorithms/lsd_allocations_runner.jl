function variable_node_iterations_lsd_optimized!(tg::LD.TannerGraph)
    @inbounds for vn_idx in 1:tg.nv
        lsd_variable_node_messages_optimized!(tg, vn_idx)
    end
    return nothing
end

function decision_step_lsd_optimized!(tg::LD.TannerGraph)
    @inbounds for vn_idx in 1:tg.nv
        lsd_variable_node_decision_optimized!(tg, vn_idx)
    end
    return nothing
end

lsd_variable_node_messages!(tg::LD.TannerGraph, vn_idx::Int64) =
    lsd_variable_node_messages_optimized!(tg, vn_idx)

_lsd_variable_node_decision!(bp_result::Vector{Float64}, tg::LD.TannerGraph, vn_idx::Int64) =
    lsd_variable_node_decision_optimized!(bp_result, tg, vn_idx)

_lsd_variable_node_decision!(tg::LD.TannerGraph, vn_idx::Int64) =
    lsd_variable_node_decision_optimized!(tg, vn_idx)

function run_belief_propagation_lsd_optimized!(
    tg::LD.TannerGraph,
    message::Vector{Float64},
    σ::Float64,
    max_iter::Int64;
    search_interval::Float64=1.5,
    lsd_beta::Float64=LatticeDecoder.LSD_DEFAULT_BETA,
    lsd_w_min::Float64=LatticeDecoder.LSD_W_MIN,
    threaded::Bool=false,
)
    _validate_lsd_beta(lsd_beta)
    _validate_lsd_w_min(lsd_w_min)
    if threaded
        throw(ArgumentError("Internal threading is disabled. Use outer-loop threading instead and call with threaded=false."))
    end

    tg.search_interval = search_interval
    tg.lsd_beta = lsd_beta
    tg.lsd_w_min = lsd_w_min
    LD.initialize_messages!(tg, message, σ)

    @inbounds for _ in 1:max_iter
        LD.check_node_iterations!(tg)
        variable_node_iterations_lsd_optimized!(tg)
    end

    decision_step_lsd_optimized!(tg)
    return tg.bp_result
end
