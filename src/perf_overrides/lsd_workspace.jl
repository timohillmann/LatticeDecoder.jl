const LSD_W_MIN = 0.95
const LSD_EPSILON = 1e-10
const LSD_MAX_ITER = 50
const LSD_DEFAULT_MAX_CANDIDATES = 128
const LSD_TLS_WORKSPACE_KEY = :lsd_perf_overlay_workspaces

mutable struct LSDWorkspace
    msg_vector::Vector{LD.gaussian_log_weight}

    t_vector::Vector{Float64}
    g_vector::Vector{Float64}
    p_vector::Vector{Float64}
    f_vector::Vector{Float64}
    r_vector::Vector{Float64}
    coeff_vector::Vector{Float64}

    dist_vector::Vector{Float64}
    z_vector::Vector{Float64}
    s_vector::Vector{Float64}
    gamma_vector::Vector{Float64}
    u_vector::Vector{Float64}

    candidate_means::Vector{Float64}
    candidate_log_weights::Vector{Float64}
    normalized_weights::Vector{Float64}

    msg_inv_var::Vector{Float64}
    msg_mean_invvar::Vector{Float64}
    msg_t_raw::Vector{Float64}
    msg_g::Vector{Float64}
    msg_p::Vector{Float64}
    msg_coeff::Vector{Float64}
    msg_beta_contrib::Vector{Float64}
    prefix_beta_max::Vector{Float64}
    suffix_beta_max::Vector{Float64}

    mean_invvar_sum::Float64
    max_n::Int
    max_candidates::Int
end

function LSDWorkspace(max_n::Int, max_candidates::Int)
    n = max(max_n, 2)
    c = max(max_candidates, 8)
    return LSDWorkspace(
        Vector{LD.gaussian_log_weight}(undef, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, c),
        zeros(Float64, c),
        zeros(Float64, c),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        fill(-Inf, n),
        fill(-Inf, n),
        fill(-Inf, n),
        0.0,
        n,
        c,
    )
end

function ensure_workspace_capacity!(ws::LSDWorkspace, required_n::Int, required_candidates::Int)
    if required_n > ws.max_n
        resize!(ws.msg_vector, required_n)
        resize!(ws.t_vector, required_n)
        resize!(ws.g_vector, required_n)
        resize!(ws.p_vector, required_n)
        resize!(ws.f_vector, required_n)
        resize!(ws.r_vector, required_n)
        resize!(ws.coeff_vector, required_n)
        resize!(ws.dist_vector, required_n)
        resize!(ws.z_vector, required_n)
        resize!(ws.s_vector, required_n)
        resize!(ws.gamma_vector, required_n)
        resize!(ws.u_vector, required_n)
        resize!(ws.msg_inv_var, required_n)
        resize!(ws.msg_mean_invvar, required_n)
        resize!(ws.msg_t_raw, required_n)
        resize!(ws.msg_g, required_n)
        resize!(ws.msg_p, required_n)
        resize!(ws.msg_coeff, required_n)
        resize!(ws.msg_beta_contrib, required_n)
        resize!(ws.prefix_beta_max, required_n)
        resize!(ws.suffix_beta_max, required_n)
        ws.max_n = required_n
    end

    if required_candidates > ws.max_candidates
        resize!(ws.candidate_means, required_candidates)
        resize!(ws.candidate_log_weights, required_candidates)
        resize!(ws.normalized_weights, required_candidates)
        ws.max_candidates = required_candidates
    end

    return ws
end

function _task_workspace_bucket()
    tls = task_local_storage()
    bucket = get(tls, LSD_TLS_WORKSPACE_KEY, nothing)
    if bucket === nothing
        bucket = Dict{Int, LSDWorkspace}()
        tls[LSD_TLS_WORKSPACE_KEY] = bucket
    end
    return bucket::Dict{Int, LSDWorkspace}
end

function get_lsd_workspace(msg_count::Int; max_candidates::Int=LSD_DEFAULT_MAX_CANDIDATES)
    bucket = _task_workspace_bucket()
    ws = get!(bucket, msg_count) do
        LSDWorkspace(msg_count, max_candidates)
    end
    ensure_workspace_capacity!(ws, msg_count, max_candidates)
    return ws
end

function clear_lsd_workspaces!()
    tls = task_local_storage()
    if haskey(tls, LSD_TLS_WORKSPACE_KEY)
        empty!(tls[LSD_TLS_WORKSPACE_KEY])
    end
    return nothing
end
