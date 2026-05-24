function build_lsd_input_excluding!(allocs::LSDAllocations, vn::LD.VariableNode, exclude_idx::Int, beta::Float64=LatticeDecoder.LSD_DEFAULT_BETA, w_min::Float64=LatticeDecoder.LSD_W_MIN)
    beta > 0 || throw(ArgumentError("LSD beta must be positive."))
    w_min >= 0 || throw(ArgumentError("LSD w_min must be nonnegative."))
    d = length(vn.messages)
    n = d

    Vinv = 0.0
    mean_invvar_sum = 0.0
    out_idx = 1

    @inbounds for i in 1:d
        if i != exclude_idx
            msg = vn.messages[i]
            inv_var = 1.0 / msg.var
            Vinv += inv_var
            mean_invvar_sum += msg.mean * inv_var

            allocs.t_vector[out_idx] = sign(msg.period) * sqrt(inv_var)
            allocs.g_vector[out_idx] = sqrt(msg.var * msg.period^2)
            allocs.p_vector[out_idx] = msg.mean * msg.period
            allocs.coeff_vector[out_idx] = inv_var / msg.period

            if abs(msg.period) < w_min
                beta = max(beta, 1.0 / sqrt(msg.var * msg.period^2))
            end
            out_idx += 1
        end
    end

    channel_msg = vn.message
    inv_var = 1.0 / channel_msg.var
    Vinv += inv_var
    mean_invvar_sum += channel_msg.mean * inv_var

    allocs.t_vector[n] = sign(channel_msg.period) * sqrt(inv_var)
    allocs.g_vector[n] = sqrt(channel_msg.var * channel_msg.period^2)
    allocs.p_vector[n] = channel_msg.mean * channel_msg.period
    allocs.coeff_vector[n] = inv_var / channel_msg.period

    if abs(channel_msg.period) < w_min
        beta = max(beta, 1.0 / sqrt(channel_msg.var * channel_msg.period^2))
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds for i in 1:n
        allocs.t_vector[i] *= inv_sqrt_vinv
    end

    @inbounds begin
        allocs.f_vector[1] = 1.0 - allocs.t_vector[1]^2
        for i in 2:n
            allocs.f_vector[i] = allocs.f_vector[i - 1] - allocs.t_vector[i]^2
        end
    end

    @inbounds begin
        allocs.r_vector[1] = (1.0 / allocs.g_vector[1]^2) * allocs.f_vector[1]
        for i in 2:(n - 1)
            allocs.r_vector[i] = (1.0 / allocs.g_vector[i]^2) * allocs.f_vector[i] / allocs.f_vector[i - 1]
        end
        allocs.r_vector[n] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = channel_msg.mean / channel_msg.var * inv_sqrt_vinv
    allocs.mean_invvar_sum = mean_invvar_sum

    return n, var, beta, beta1, u_d
end

function build_lsd_input_all!(allocs::LSDAllocations, vn::LD.VariableNode, beta::Float64=LatticeDecoder.LSD_DEFAULT_BETA, w_min::Float64=LatticeDecoder.LSD_W_MIN)
    beta > 0 || throw(ArgumentError("LSD beta must be positive."))
    w_min >= 0 || throw(ArgumentError("LSD w_min must be nonnegative."))
    d = length(vn.messages)
    n = d + 1

    Vinv = 0.0
    mean_invvar_sum = 0.0

    @inbounds for i in 1:d
        msg = vn.messages[i]
        inv_var = 1.0 / msg.var
        Vinv += inv_var
        mean_invvar_sum += msg.mean * inv_var

        allocs.t_vector[i] = sign(msg.period) * sqrt(inv_var)
        allocs.g_vector[i] = sqrt(msg.var * msg.period^2)
        allocs.p_vector[i] = msg.mean * msg.period
        allocs.coeff_vector[i] = inv_var / msg.period

        if abs(msg.period) < w_min
            beta = max(beta, 1.0 / sqrt(msg.var * msg.period^2))
        end
    end

    channel_msg = vn.message
    inv_var = 1.0 / channel_msg.var
    Vinv += inv_var
    mean_invvar_sum += channel_msg.mean * inv_var

    allocs.t_vector[n] = sign(channel_msg.period) * sqrt(inv_var)
    allocs.g_vector[n] = sqrt(channel_msg.var * channel_msg.period^2)
    allocs.p_vector[n] = channel_msg.mean * channel_msg.period
    allocs.coeff_vector[n] = inv_var / channel_msg.period

    if abs(channel_msg.period) < w_min
        beta = max(beta, 1.0 / sqrt(channel_msg.var * channel_msg.period^2))
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds for i in 1:n
        allocs.t_vector[i] *= inv_sqrt_vinv
    end

    @inbounds begin
        allocs.f_vector[1] = 1.0 - allocs.t_vector[1]^2
        for i in 2:n
            allocs.f_vector[i] = allocs.f_vector[i - 1] - allocs.t_vector[i]^2
        end
    end

    @inbounds begin
        allocs.r_vector[1] = (1.0 / allocs.g_vector[1]^2) * allocs.f_vector[1]
        for i in 2:(n - 1)
            allocs.r_vector[i] = (1.0 / allocs.g_vector[i]^2) * allocs.f_vector[i] / allocs.f_vector[i - 1]
        end
        allocs.r_vector[n] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = channel_msg.mean / channel_msg.var * inv_sqrt_vinv
    allocs.mean_invvar_sum = mean_invvar_sum

    return n, var, beta, beta1, u_d
end

@inline function _append_candidate!(allocs::LSDAllocations, count::Int, dist_k::Float64, var::Float64, sum_az::Float64)
    required = count + 1
    if required > allocs.max_candidates
        return count
    end

    allocs.candidate_means[required] = var * (allocs.mean_invvar_sum + sum_az)
    allocs.candidate_log_weights[required] = -0.5 * dist_k
    return required
end

@inline _lsd_overlay_schnorr_euchner_step(delta::Float64) = delta < 0 ? -1.0 : 1.0

function simplified_lsd_candidates!(allocs::LSDAllocations, n::Int, var::Float64, beta::Float64, beta1::Float64, u_d::Float64)
    dist = allocs.dist_vector
    z = allocs.z_vector
    s = allocs.s_vector
    gamma = allocs.gamma_vector
    u = allocs.u_vector
    p = allocs.p_vector
    t = allocs.t_vector
    g = allocs.g_vector
    f = allocs.f_vector
    r = allocs.r_vector
    coeff = allocs.coeff_vector

    @inbounds for i in 1:n
        dist[i] = 0.0
        z[i] = 0.0
        s[i] = 0.0
        gamma[i] = 0.0
        u[i] = 0.0
    end

    d = n
    k = d - 1
    if k < 1
        return 0
    end

    u[d] = u_d

    @inbounds begin
        gamma[k] = -p[k] + t[k] * g[k] * u[k + 1] / f[k]
        z[k] = round(gamma[k])
        s[k] = _lsd_overlay_schnorr_euchner_step(gamma[k] - z[k])
        dist[k] = dist[k + 1] + (gamma[k] - z[k])^2 * r[k]
    end

    sum_az = coeff[k] * z[k]
    count = 0
    beta_sq = beta^2

    while k <= (d - 1)
        if dist[k] <= beta_sq
            if k == 1
                next_count = _append_candidate!(allocs, count, dist[k], var, sum_az)
                next_count == count && return count
                count = next_count
                if count == 1
                    beta = min(beta1, sqrt(dist[k] - 2.0 * log(LatticeDecoder.LSD_EPSILON)))
                    beta_sq = beta^2
                end

                @inbounds begin
                    old_z = z[k]
                    z[k] += s[k]
                    sum_az += coeff[k] * (z[k] - old_z)
                    s[k] = -s[k] - sign(s[k])
                    dist[k] = dist[k + 1] + (gamma[k] - z[k])^2 * r[k]
                end
            else
                @inbounds begin
                    u[k] = t[k] * (z[k] + p[k]) / g[k] + u[k + 1]
                    k -= 1

                    gamma[k] = -p[k] + t[k] * g[k] * u[k + 1] / f[k]
                    old_z = z[k]
                    z[k] = round(gamma[k])
                    sum_az += coeff[k] * (z[k] - old_z)
                    s[k] = _lsd_overlay_schnorr_euchner_step(gamma[k] - z[k])
                    dist[k] = dist[k + 1] + (gamma[k] - z[k])^2 * r[k]
                end
            end
        else
            if k == (d - 1)
                return count
            else
                @inbounds begin
                    k += 1
                    old_z = z[k]
                    z[k] += s[k]
                    sum_az += coeff[k] * (z[k] - old_z)
                    s[k] = -s[k] - sign(s[k])
                    dist[k] = dist[k + 1] + (gamma[k] - z[k])^2 * r[k]
                end
            end
        end
    end

    return count
end

function moment_match_candidates!(out::LD.gaussian_log_weight, allocs::LSDAllocations, count::Int, var::Float64)
    if count == 0
        return false
    end

    max_logw = -Inf
    @inbounds for i in 1:count
        max_logw = max(max_logw, allocs.candidate_log_weights[i])
    end

    weight_sum = 0.0
    @inbounds for i in 1:count
        w = exp(allocs.candidate_log_weights[i] - max_logw)
        allocs.normalized_weights[i] = w
        weight_sum += w
    end

    inv_weight_sum = 1.0 / weight_sum

    mean = 0.0
    second_moment = 0.0
    @inbounds for i in 1:count
        w = allocs.normalized_weights[i] * inv_weight_sum
        m = allocs.candidate_means[i]
        mean += w * m
        second_moment += w * (var + m^2)
    end

    variance = second_moment - mean^2
    out.mean = mean
    out.var = max(variance, LatticeDecoder.MIN_VAR)
    return true
end

function lsd_variable_node_message_optimized!(cn_message::LD.gaussian_log_weight, vn::LD.VariableNode, nb_idx::Int, beta::Float64=LatticeDecoder.LSD_DEFAULT_BETA, w_min::Float64=LatticeDecoder.LSD_W_MIN)
    d = length(vn.messages)
    allocs = get_lsd_allocations(d)

    if d == 1
        @inbounds begin
            cn_message.mean = vn.message.mean
            cn_message.var = vn.message.var
            cn_message.period = vn.message.period
        end
        return 1
    end

    n, var, beta, beta1, u_d = build_lsd_input_excluding!(allocs, vn, nb_idx, beta, w_min)
    count = simplified_lsd_candidates!(allocs, n, var, beta, beta1, u_d)
    if count == 0
        return 0
    end

    moment_match_candidates!(cn_message, allocs, count, var)
    return count
end

function lsd_variable_node_messages_optimized!(tg::LD.TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    d = length(vn.messages)
    allocs = get_lsd_allocations(d)

    if d == 1
        channel_msg = vn.message
        @inbounds for j in 1:length(vn.neighbours)
            cn_idx, _ = vn.neighbours[j]
            idx = vn.pos_in_check_neighbour[j]
            cn = tg.check_nodes[cn_idx]
            cn.messages[idx].mean = channel_msg.mean
            cn.messages[idx].var = channel_msg.var
            cn.messages[idx].period = channel_msg.period
        end
        return nothing
    end

    @inbounds for j in 1:length(vn.neighbours)
        cn_idx, _ = vn.neighbours[j]
        idx = vn.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]

        n, var, beta, beta1, u_d = build_lsd_input_excluding!(allocs, vn, j, tg.lsd_beta, tg.lsd_w_min)
        count = simplified_lsd_candidates!(allocs, n, var, beta, beta1, u_d)
        if count > 0
            moment_match_candidates!(cn.messages[idx], allocs, count, var)
        end
    end
    return nothing
end

function lsd_variable_node_decision_optimized!(bp_result::Vector{Float64}, tg::LD.TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    d = length(vn.messages)
    n = d + 1
    allocs = get_lsd_allocations(n)

    if n == 2
        @inbounds begin
            msg1 = vn.messages[1]
            if isapprox(msg1.var, LatticeDecoder.MIN_VAR)
                bp_result[vn_idx] = msg1.mean
                return 1
            end
            msg2 = vn.message
            if isapprox(msg2.var, LatticeDecoder.MIN_VAR)
                bp_result[vn_idx] = msg2.mean
                return 1
            end
        end
    end

    n, var, beta, beta1, u_d = build_lsd_input_all!(allocs, vn, tg.lsd_beta, tg.lsd_w_min)
    count = simplified_lsd_candidates!(allocs, n, var, beta, beta1, u_d)
    if count > 0
        moment_match_candidates!(vn.message, allocs, count, var)
    end

    bp_result[vn_idx] = vn.message.mean
    return count
end

lsd_variable_node_decision_optimized!(tg::LD.TannerGraph, vn_idx::Int64) =
    lsd_variable_node_decision_optimized!(tg.bp_result, tg, vn_idx)
