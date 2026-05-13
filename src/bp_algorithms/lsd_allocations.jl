mutable struct LSDAllocations
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

    mean_invvar_sum::Float64
    max_n::Int
    max_candidates::Int
end

function LSDAllocations(max_n::Int, max_candidates::Int)
    n = max(max_n, 2)
    c = max(max_candidates, 8)
    return LSDAllocations(
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
        0.0,
        n,
        c,
    )
end

function ensure_allocations_capacity!(allocs::LSDAllocations, required_n::Int, required_candidates::Int)
    if required_n > allocs.max_n
        resize!(allocs.t_vector, required_n)
        resize!(allocs.g_vector, required_n)
        resize!(allocs.p_vector, required_n)
        resize!(allocs.f_vector, required_n)
        resize!(allocs.r_vector, required_n)
        resize!(allocs.coeff_vector, required_n)
        resize!(allocs.dist_vector, required_n)
        resize!(allocs.z_vector, required_n)
        resize!(allocs.s_vector, required_n)
        resize!(allocs.gamma_vector, required_n)
        resize!(allocs.u_vector, required_n)
        allocs.max_n = required_n
    end

    if required_candidates > allocs.max_candidates
        resize!(allocs.candidate_means, required_candidates)
        resize!(allocs.candidate_log_weights, required_candidates)
        resize!(allocs.normalized_weights, required_candidates)
        allocs.max_candidates = required_candidates
    end

    return allocs
end

function _task_allocations_bucket()
    tls = task_local_storage()
    bucket = get(tls, LatticeDecoder.LSD_TLS_ALLOCATIONS_KEY, nothing)
    if bucket === nothing
        bucket = Dict{Int, LSDAllocations}()
        tls[LatticeDecoder.LSD_TLS_ALLOCATIONS_KEY] = bucket
    end
    return bucket::Dict{Int, LSDAllocations}
end

function get_lsd_allocations(msg_count::Int; max_candidates::Int=LatticeDecoder.LSD_DEFAULT_MAX_CANDIDATES)
    bucket = _task_allocations_bucket()
    allocs = get!(bucket, msg_count) do
        LSDAllocations(msg_count, max_candidates)
    end
    ensure_allocations_capacity!(allocs, msg_count, max_candidates)
    return allocs
end

function clear_lsd_allocations!()
    tls = task_local_storage()
    if haskey(tls, LatticeDecoder.LSD_TLS_ALLOCATIONS_KEY)
        empty!(tls[LatticeDecoder.LSD_TLS_ALLOCATIONS_KEY])
    end
    return nothing
end
