const EXCL_MAP_D5 = (
    (2, 3, 4, 5),
    (1, 3, 4, 5),
    (1, 2, 4, 5),
    (1, 2, 3, 5),
    (1, 2, 3, 4),
)

const EXCL_MAP_D7 = (
    (2, 3, 4, 5, 6, 7),
    (1, 3, 4, 5, 6, 7),
    (1, 2, 4, 5, 6, 7),
    (1, 2, 3, 5, 6, 7),
    (1, 2, 3, 4, 6, 7),
    (1, 2, 3, 4, 5, 7),
    (1, 2, 3, 4, 5, 6),
)

@inline function collect_msg_vector_excluding!(ws::LSDWorkspace, vn::LD.VariableNode, exclude_idx::Int)
    d = length(vn.messages)
    idx = 1
    @inbounds for i in 1:d
        if i != exclude_idx
            ws.msg_vector[idx] = vn.messages[i]
            idx += 1
        end
    end
    @inbounds ws.msg_vector[idx] = vn.message
    return d
end

function build_lsd_input_excluding_precomputed_d5!(
    ws::LSDWorkspace,
    exclude_idx::Int,
    total_inv_var::Float64,
    total_mean_invvar::Float64,
    channel_inv_var::Float64,
    channel_mean_invvar::Float64,
    channel_t_raw::Float64,
    channel_g::Float64,
    channel_p::Float64,
    channel_coeff::Float64,
    channel_beta_contrib::Float64,
)
    Vinv = total_inv_var - ws.msg_inv_var[exclude_idx] + channel_inv_var
    mean_invvar_sum = total_mean_invvar - ws.msg_mean_invvar[exclude_idx] + channel_mean_invvar

    beta_msg = _beta_max_excluding(ws, exclude_idx, 5)
    beta = max(max(3.5, channel_beta_contrib), beta_msg)

    i1, i2, i3, i4 = EXCL_MAP_D5[exclude_idx]
    @inbounds begin
        ws.t_vector[1] = ws.msg_t_raw[i1]
        ws.t_vector[2] = ws.msg_t_raw[i2]
        ws.t_vector[3] = ws.msg_t_raw[i3]
        ws.t_vector[4] = ws.msg_t_raw[i4]
        ws.t_vector[5] = channel_t_raw

        ws.g_vector[1] = ws.msg_g[i1]
        ws.g_vector[2] = ws.msg_g[i2]
        ws.g_vector[3] = ws.msg_g[i3]
        ws.g_vector[4] = ws.msg_g[i4]
        ws.g_vector[5] = channel_g

        ws.p_vector[1] = ws.msg_p[i1]
        ws.p_vector[2] = ws.msg_p[i2]
        ws.p_vector[3] = ws.msg_p[i3]
        ws.p_vector[4] = ws.msg_p[i4]
        ws.p_vector[5] = channel_p

        ws.coeff_vector[1] = ws.msg_coeff[i1]
        ws.coeff_vector[2] = ws.msg_coeff[i2]
        ws.coeff_vector[3] = ws.msg_coeff[i3]
        ws.coeff_vector[4] = ws.msg_coeff[i4]
        ws.coeff_vector[5] = channel_coeff
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds begin
        t1 = ws.t_vector[1] * inv_sqrt_vinv
        t2 = ws.t_vector[2] * inv_sqrt_vinv
        t3 = ws.t_vector[3] * inv_sqrt_vinv
        t4 = ws.t_vector[4] * inv_sqrt_vinv
        t5 = ws.t_vector[5] * inv_sqrt_vinv
        ws.t_vector[1] = t1
        ws.t_vector[2] = t2
        ws.t_vector[3] = t3
        ws.t_vector[4] = t4
        ws.t_vector[5] = t5

        f1 = 1.0 - t1^2
        f2 = f1 - t2^2
        f3 = f2 - t3^2
        f4 = f3 - t4^2
        f5 = f4 - t5^2
        ws.f_vector[1] = f1
        ws.f_vector[2] = f2
        ws.f_vector[3] = f3
        ws.f_vector[4] = f4
        ws.f_vector[5] = f5

        g1 = ws.g_vector[1]
        g2 = ws.g_vector[2]
        g3 = ws.g_vector[3]
        g4 = ws.g_vector[4]
        ws.r_vector[1] = (1.0 / g1^2) * f1
        ws.r_vector[2] = (1.0 / g2^2) * f2 / f1
        ws.r_vector[3] = (1.0 / g3^2) * f3 / f2
        ws.r_vector[4] = (1.0 / g4^2) * f4 / f3
        ws.r_vector[5] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = channel_mean_invvar * inv_sqrt_vinv
    ws.mean_invvar_sum = mean_invvar_sum

    return 5, var, beta, beta1, u_d
end

function build_lsd_input_excluding_precomputed_d7!(
    ws::LSDWorkspace,
    exclude_idx::Int,
    total_inv_var::Float64,
    total_mean_invvar::Float64,
    channel_inv_var::Float64,
    channel_mean_invvar::Float64,
    channel_t_raw::Float64,
    channel_g::Float64,
    channel_p::Float64,
    channel_coeff::Float64,
    channel_beta_contrib::Float64,
)
    Vinv = total_inv_var - ws.msg_inv_var[exclude_idx] + channel_inv_var
    mean_invvar_sum = total_mean_invvar - ws.msg_mean_invvar[exclude_idx] + channel_mean_invvar

    beta_msg = _beta_max_excluding(ws, exclude_idx, 7)
    beta = max(max(3.5, channel_beta_contrib), beta_msg)

    i1, i2, i3, i4, i5, i6 = EXCL_MAP_D7[exclude_idx]
    @inbounds begin
        ws.t_vector[1] = ws.msg_t_raw[i1]
        ws.t_vector[2] = ws.msg_t_raw[i2]
        ws.t_vector[3] = ws.msg_t_raw[i3]
        ws.t_vector[4] = ws.msg_t_raw[i4]
        ws.t_vector[5] = ws.msg_t_raw[i5]
        ws.t_vector[6] = ws.msg_t_raw[i6]
        ws.t_vector[7] = channel_t_raw

        ws.g_vector[1] = ws.msg_g[i1]
        ws.g_vector[2] = ws.msg_g[i2]
        ws.g_vector[3] = ws.msg_g[i3]
        ws.g_vector[4] = ws.msg_g[i4]
        ws.g_vector[5] = ws.msg_g[i5]
        ws.g_vector[6] = ws.msg_g[i6]
        ws.g_vector[7] = channel_g

        ws.p_vector[1] = ws.msg_p[i1]
        ws.p_vector[2] = ws.msg_p[i2]
        ws.p_vector[3] = ws.msg_p[i3]
        ws.p_vector[4] = ws.msg_p[i4]
        ws.p_vector[5] = ws.msg_p[i5]
        ws.p_vector[6] = ws.msg_p[i6]
        ws.p_vector[7] = channel_p

        ws.coeff_vector[1] = ws.msg_coeff[i1]
        ws.coeff_vector[2] = ws.msg_coeff[i2]
        ws.coeff_vector[3] = ws.msg_coeff[i3]
        ws.coeff_vector[4] = ws.msg_coeff[i4]
        ws.coeff_vector[5] = ws.msg_coeff[i5]
        ws.coeff_vector[6] = ws.msg_coeff[i6]
        ws.coeff_vector[7] = channel_coeff
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds begin
        t1 = ws.t_vector[1] * inv_sqrt_vinv
        t2 = ws.t_vector[2] * inv_sqrt_vinv
        t3 = ws.t_vector[3] * inv_sqrt_vinv
        t4 = ws.t_vector[4] * inv_sqrt_vinv
        t5 = ws.t_vector[5] * inv_sqrt_vinv
        t6 = ws.t_vector[6] * inv_sqrt_vinv
        t7 = ws.t_vector[7] * inv_sqrt_vinv
        ws.t_vector[1] = t1
        ws.t_vector[2] = t2
        ws.t_vector[3] = t3
        ws.t_vector[4] = t4
        ws.t_vector[5] = t5
        ws.t_vector[6] = t6
        ws.t_vector[7] = t7

        f1 = 1.0 - t1^2
        f2 = f1 - t2^2
        f3 = f2 - t3^2
        f4 = f3 - t4^2
        f5 = f4 - t5^2
        f6 = f5 - t6^2
        f7 = f6 - t7^2
        ws.f_vector[1] = f1
        ws.f_vector[2] = f2
        ws.f_vector[3] = f3
        ws.f_vector[4] = f4
        ws.f_vector[5] = f5
        ws.f_vector[6] = f6
        ws.f_vector[7] = f7

        g1 = ws.g_vector[1]
        g2 = ws.g_vector[2]
        g3 = ws.g_vector[3]
        g4 = ws.g_vector[4]
        g5 = ws.g_vector[5]
        g6 = ws.g_vector[6]
        ws.r_vector[1] = (1.0 / g1^2) * f1
        ws.r_vector[2] = (1.0 / g2^2) * f2 / f1
        ws.r_vector[3] = (1.0 / g3^2) * f3 / f2
        ws.r_vector[4] = (1.0 / g4^2) * f4 / f3
        ws.r_vector[5] = (1.0 / g5^2) * f5 / f4
        ws.r_vector[6] = (1.0 / g6^2) * f6 / f5
        ws.r_vector[7] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = channel_mean_invvar * inv_sqrt_vinv
    ws.mean_invvar_sum = mean_invvar_sum

    return 7, var, beta, beta1, u_d
end

function simplified_lsd_candidates_d5!(ws::LSDWorkspace, var::Float64, beta::Float64, beta1::Float64, u_d::Float64)
    return simplified_lsd_candidates!(ws, 5, var, beta, beta1, u_d)
end

function simplified_lsd_candidates_d7!(ws::LSDWorkspace, var::Float64, beta::Float64, beta1::Float64, u_d::Float64)
    return simplified_lsd_candidates!(ws, 7, var, beta, beta1, u_d)
end

@inline function _channel_terms(msg::LD.gaussian_log_weight)
    inv_var = 1.0 / msg.var
    g = sqrt(msg.var * msg.period^2)
    beta_contrib = abs(msg.period) < LSD_W_MIN ? (1.0 / g) : -Inf
    return (
        inv_var,
        msg.mean * inv_var,
        sign(msg.period) * sqrt(inv_var),
        g,
        msg.mean * msg.period,
        inv_var / msg.period,
        beta_contrib,
    )
end

function precompute_vn_lsd_terms!(ws::LSDWorkspace, vn::LD.VariableNode)
    d = length(vn.messages)
    total_inv_var = 0.0
    total_mean_invvar = 0.0

    @inbounds for i in 1:d
        msg = vn.messages[i]
        inv_var = 1.0 / msg.var
        g = sqrt(msg.var * msg.period^2)
        beta_contrib = abs(msg.period) < LSD_W_MIN ? (1.0 / g) : -Inf

        ws.msg_inv_var[i] = inv_var
        ws.msg_mean_invvar[i] = msg.mean * inv_var
        ws.msg_t_raw[i] = sign(msg.period) * sqrt(inv_var)
        ws.msg_g[i] = g
        ws.msg_p[i] = msg.mean * msg.period
        ws.msg_coeff[i] = inv_var / msg.period
        ws.msg_beta_contrib[i] = beta_contrib

        total_inv_var += inv_var
        total_mean_invvar += ws.msg_mean_invvar[i]
    end

    if d >= 1
        @inbounds begin
            ws.prefix_beta_max[1] = ws.msg_beta_contrib[1]
            for i in 2:d
                ws.prefix_beta_max[i] = max(ws.prefix_beta_max[i - 1], ws.msg_beta_contrib[i])
            end

            ws.suffix_beta_max[d] = ws.msg_beta_contrib[d]
            for i in (d - 1):-1:1
                ws.suffix_beta_max[i] = max(ws.suffix_beta_max[i + 1], ws.msg_beta_contrib[i])
            end
        end
    end

    return total_inv_var, total_mean_invvar
end

@inline function _beta_max_excluding(ws::LSDWorkspace, exclude_idx::Int, d::Int)
    left = exclude_idx > 1 ? ws.prefix_beta_max[exclude_idx - 1] : -Inf
    right = exclude_idx < d ? ws.suffix_beta_max[exclude_idx + 1] : -Inf
    return max(left, right)
end

function build_lsd_input_excluding_precomputed!(
    ws::LSDWorkspace,
    d::Int,
    exclude_idx::Int,
    total_inv_var::Float64,
    total_mean_invvar::Float64,
    channel_inv_var::Float64,
    channel_mean_invvar::Float64,
    channel_t_raw::Float64,
    channel_g::Float64,
    channel_p::Float64,
    channel_coeff::Float64,
    channel_beta_contrib::Float64,
)
    n = d

    Vinv = total_inv_var - ws.msg_inv_var[exclude_idx] + channel_inv_var
    mean_invvar_sum = total_mean_invvar - ws.msg_mean_invvar[exclude_idx] + channel_mean_invvar

    beta_msg = _beta_max_excluding(ws, exclude_idx, d)
    beta = max(max(3.5, channel_beta_contrib), beta_msg)

    out_idx = 1
    @inbounds begin
        for i in 1:(exclude_idx - 1)
            ws.t_vector[out_idx] = ws.msg_t_raw[i]
            ws.g_vector[out_idx] = ws.msg_g[i]
            ws.p_vector[out_idx] = ws.msg_p[i]
            ws.coeff_vector[out_idx] = ws.msg_coeff[i]
            out_idx += 1
        end
        for i in (exclude_idx + 1):d
            ws.t_vector[out_idx] = ws.msg_t_raw[i]
            ws.g_vector[out_idx] = ws.msg_g[i]
            ws.p_vector[out_idx] = ws.msg_p[i]
            ws.coeff_vector[out_idx] = ws.msg_coeff[i]
            out_idx += 1
        end

        ws.t_vector[n] = channel_t_raw
        ws.g_vector[n] = channel_g
        ws.p_vector[n] = channel_p
        ws.coeff_vector[n] = channel_coeff
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds for i in 1:n
        ws.t_vector[i] *= inv_sqrt_vinv
    end

    @inbounds begin
        ws.f_vector[1] = 1.0 - ws.t_vector[1]^2
        for i in 2:n
            ws.f_vector[i] = ws.f_vector[i - 1] - ws.t_vector[i]^2
        end
    end

    @inbounds begin
        ws.r_vector[1] = (1.0 / ws.g_vector[1]^2) * ws.f_vector[1]
        for i in 2:(n - 1)
            ws.r_vector[i] = (1.0 / ws.g_vector[i]^2) * ws.f_vector[i] / ws.f_vector[i - 1]
        end
        ws.r_vector[n] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = channel_mean_invvar * inv_sqrt_vinv
    ws.mean_invvar_sum = mean_invvar_sum

    return n, var, beta, beta1, u_d
end

@inline function collect_msg_vector_all!(ws::LSDWorkspace, vn::LD.VariableNode)
    d = length(vn.messages)
    @inbounds for i in 1:d
        ws.msg_vector[i] = vn.messages[i]
    end
    @inbounds ws.msg_vector[d + 1] = vn.message
    return d + 1
end

function build_lsd_input!(ws::LSDWorkspace, n::Int)
    Vinv = 0.0
    beta = 3.5
    mean_invvar_sum = 0.0

    @inbounds for i in 1:n
        msg = ws.msg_vector[i]
        inv_var = 1.0 / msg.var
        Vinv += inv_var
        mean_invvar_sum += msg.mean * inv_var

        ws.t_vector[i] = sign(msg.period) * sqrt(inv_var)
        ws.g_vector[i] = sqrt(msg.var * msg.period^2)
        ws.p_vector[i] = msg.mean * msg.period
        ws.coeff_vector[i] = inv_var / msg.period

        if abs(msg.period) < LSD_W_MIN
            beta = max(beta, 1.0 / sqrt(msg.var * msg.period^2))
        end
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds for i in 1:n
        ws.t_vector[i] *= inv_sqrt_vinv
    end

    @inbounds begin
        ws.f_vector[1] = 1.0 - ws.t_vector[1]^2
        for i in 2:n
            ws.f_vector[i] = ws.f_vector[i - 1] - ws.t_vector[i]^2
        end
    end

    @inbounds begin
        ws.r_vector[1] = (1.0 / ws.g_vector[1]^2) * ws.f_vector[1]
        for i in 2:(n - 1)
            ws.r_vector[i] = (1.0 / ws.g_vector[i]^2) * ws.f_vector[i] / ws.f_vector[i - 1]
        end
        ws.r_vector[n] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = ws.msg_vector[n].mean / ws.msg_vector[n].var * inv_sqrt_vinv
    ws.mean_invvar_sum = mean_invvar_sum

    return var, beta, beta1, u_d
end

function build_lsd_input_excluding!(ws::LSDWorkspace, vn::LD.VariableNode, exclude_idx::Int)
    d = length(vn.messages)
    n = d

    Vinv = 0.0
    beta = 3.5
    mean_invvar_sum = 0.0
    out_idx = 1

    @inbounds for i in 1:d
        if i != exclude_idx
            msg = vn.messages[i]
            inv_var = 1.0 / msg.var
            Vinv += inv_var
            mean_invvar_sum += msg.mean * inv_var

            ws.t_vector[out_idx] = sign(msg.period) * sqrt(inv_var)
            ws.g_vector[out_idx] = sqrt(msg.var * msg.period^2)
            ws.p_vector[out_idx] = msg.mean * msg.period
            ws.coeff_vector[out_idx] = inv_var / msg.period

            if abs(msg.period) < LSD_W_MIN
                beta = max(beta, 1.0 / sqrt(msg.var * msg.period^2))
            end
            out_idx += 1
        end
    end

    channel_msg = vn.message
    inv_var = 1.0 / channel_msg.var
    Vinv += inv_var
    mean_invvar_sum += channel_msg.mean * inv_var

    ws.t_vector[n] = sign(channel_msg.period) * sqrt(inv_var)
    ws.g_vector[n] = sqrt(channel_msg.var * channel_msg.period^2)
    ws.p_vector[n] = channel_msg.mean * channel_msg.period
    ws.coeff_vector[n] = inv_var / channel_msg.period

    if abs(channel_msg.period) < LSD_W_MIN
        beta = max(beta, 1.0 / sqrt(channel_msg.var * channel_msg.period^2))
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds for i in 1:n
        ws.t_vector[i] *= inv_sqrt_vinv
    end

    @inbounds begin
        ws.f_vector[1] = 1.0 - ws.t_vector[1]^2
        for i in 2:n
            ws.f_vector[i] = ws.f_vector[i - 1] - ws.t_vector[i]^2
        end
    end

    @inbounds begin
        ws.r_vector[1] = (1.0 / ws.g_vector[1]^2) * ws.f_vector[1]
        for i in 2:(n - 1)
            ws.r_vector[i] = (1.0 / ws.g_vector[i]^2) * ws.f_vector[i] / ws.f_vector[i - 1]
        end
        ws.r_vector[n] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = channel_msg.mean / channel_msg.var * inv_sqrt_vinv
    ws.mean_invvar_sum = mean_invvar_sum

    return n, var, beta, beta1, u_d
end

function build_lsd_input_all!(ws::LSDWorkspace, vn::LD.VariableNode)
    d = length(vn.messages)
    n = d + 1

    Vinv = 0.0
    beta = 3.5
    mean_invvar_sum = 0.0

    @inbounds for i in 1:d
        msg = vn.messages[i]
        inv_var = 1.0 / msg.var
        Vinv += inv_var
        mean_invvar_sum += msg.mean * inv_var

        ws.t_vector[i] = sign(msg.period) * sqrt(inv_var)
        ws.g_vector[i] = sqrt(msg.var * msg.period^2)
        ws.p_vector[i] = msg.mean * msg.period
        ws.coeff_vector[i] = inv_var / msg.period

        if abs(msg.period) < LSD_W_MIN
            beta = max(beta, 1.0 / sqrt(msg.var * msg.period^2))
        end
    end

    channel_msg = vn.message
    inv_var = 1.0 / channel_msg.var
    Vinv += inv_var
    mean_invvar_sum += channel_msg.mean * inv_var

    ws.t_vector[n] = sign(channel_msg.period) * sqrt(inv_var)
    ws.g_vector[n] = sqrt(channel_msg.var * channel_msg.period^2)
    ws.p_vector[n] = channel_msg.mean * channel_msg.period
    ws.coeff_vector[n] = inv_var / channel_msg.period

    if abs(channel_msg.period) < LSD_W_MIN
        beta = max(beta, 1.0 / sqrt(channel_msg.var * channel_msg.period^2))
    end

    inv_sqrt_vinv = 1.0 / sqrt(Vinv)
    @inbounds for i in 1:n
        ws.t_vector[i] *= inv_sqrt_vinv
    end

    @inbounds begin
        ws.f_vector[1] = 1.0 - ws.t_vector[1]^2
        for i in 2:n
            ws.f_vector[i] = ws.f_vector[i - 1] - ws.t_vector[i]^2
        end
    end

    @inbounds begin
        ws.r_vector[1] = (1.0 / ws.g_vector[1]^2) * ws.f_vector[1]
        for i in 2:(n - 1)
            ws.r_vector[i] = (1.0 / ws.g_vector[i]^2) * ws.f_vector[i] / ws.f_vector[i - 1]
        end
        ws.r_vector[n] = 0.0
    end

    var = 1.0 / Vinv
    beta1 = beta
    u_d = channel_msg.mean / channel_msg.var * inv_sqrt_vinv
    ws.mean_invvar_sum = mean_invvar_sum

    return n, var, beta, beta1, u_d
end

@inline function _append_candidate!(ws::LSDWorkspace, count::Int, dist_k::Float64, var::Float64, sum_az::Float64)
    required = count + 1
    if required > ws.max_candidates
        ensure_workspace_capacity!(ws, ws.max_n, max(required, ws.max_candidates * 2))
    end

    ws.candidate_means[required] = var * (ws.mean_invvar_sum + sum_az)
    ws.candidate_log_weights[required] = -0.5 * dist_k
    return required
end

function simplified_lsd_candidates!(ws::LSDWorkspace, n::Int, var::Float64, beta::Float64, beta1::Float64, u_d::Float64)
    dist = ws.dist_vector
    z = ws.z_vector
    s = ws.s_vector
    gamma = ws.gamma_vector
    u = ws.u_vector
    p = ws.p_vector
    t = ws.t_vector
    g = ws.g_vector
    f = ws.f_vector
    r = ws.r_vector
    coeff = ws.coeff_vector

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
        s[k] = sign(gamma[k] - z[k])
        dist[k] = dist[k + 1] + (gamma[k] - z[k])^2 * r[k]
    end

    sum_az = coeff[k] * z[k]
    count = 0
    iter = 0
    beta_sq = beta^2

    while k <= (d - 1) && iter <= LSD_MAX_ITER
        iter += 1
        if dist[k] <= beta_sq
            if k == 1
                count = _append_candidate!(ws, count, dist[k], var, sum_az)
                if count == 1
                    beta = min(beta1, sqrt(dist[k] - 2.0 * log(LSD_EPSILON)))
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
                    s[k] = sign(gamma[k] - z[k])
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

function moment_match_candidates!(out::LD.gaussian_log_weight, ws::LSDWorkspace, count::Int, var::Float64)
    if count == 0
        return false
    end

    max_logw = -Inf
    @inbounds for i in 1:count
        max_logw = max(max_logw, ws.candidate_log_weights[i])
    end

    weight_sum = 0.0
    @inbounds for i in 1:count
        w = exp(ws.candidate_log_weights[i] - max_logw)
        ws.normalized_weights[i] = w
        weight_sum += w
    end

    inv_weight_sum = 1.0 / weight_sum

    mean = 0.0
    second_moment = 0.0
    @inbounds for i in 1:count
        w = ws.normalized_weights[i] * inv_weight_sum
        m = ws.candidate_means[i]
        mean += w * m
        second_moment += w * (var + m^2)
    end

    variance = second_moment - mean^2
    out.mean = mean
    out.var = max(variance, LD.MIN_VAR)
    return true
end

function lsd_variable_node_message_optimized!(cn_message::LD.gaussian_log_weight, vn::LD.VariableNode, nb_idx::Int)
    d = length(vn.messages)
    ws = get_lsd_workspace(d)

    if d == 1
        @inbounds begin
            cn_message.mean = vn.message.mean
            cn_message.var = vn.message.var
            cn_message.period = vn.message.period
        end
        return 1
    end

    total_inv_var, total_mean_invvar = precompute_vn_lsd_terms!(ws, vn)
    channel_inv_var, channel_mean_invvar, channel_t_raw, channel_g, channel_p, channel_coeff, channel_beta_contrib =
        _channel_terms(vn.message)

    n, var, beta, beta1, u_d = build_lsd_input_excluding_precomputed!(
        ws,
        d,
        nb_idx,
        total_inv_var,
        total_mean_invvar,
        channel_inv_var,
        channel_mean_invvar,
        channel_t_raw,
        channel_g,
        channel_p,
        channel_coeff,
        channel_beta_contrib,
    )
    count = simplified_lsd_candidates!(ws, n, var, beta, beta1, u_d)
    if count == 0
        return 0
    end

    moment_match_candidates!(cn_message, ws, count, var)
    return count
end

function lsd_variable_node_messages_optimized!(tg::LD.TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    d = length(vn.messages)
    ws = get_lsd_workspace(d)

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

    total_inv_var, total_mean_invvar = precompute_vn_lsd_terms!(ws, vn)
    channel_inv_var, channel_mean_invvar, channel_t_raw, channel_g, channel_p, channel_coeff, channel_beta_contrib =
        _channel_terms(vn.message)

    if d == 5
        @inbounds for j in 1:length(vn.neighbours)
            cn_idx, _ = vn.neighbours[j]
            idx = vn.pos_in_check_neighbour[j]
            cn = tg.check_nodes[cn_idx]

            _, var, beta, beta1, u_d = build_lsd_input_excluding_precomputed_d5!(
                ws,
                j,
                total_inv_var,
                total_mean_invvar,
                channel_inv_var,
                channel_mean_invvar,
                channel_t_raw,
                channel_g,
                channel_p,
                channel_coeff,
                channel_beta_contrib,
            )
            count = simplified_lsd_candidates_d5!(ws, var, beta, beta1, u_d)
            if count > 0
                moment_match_candidates!(cn.messages[idx], ws, count, var)
            end
        end
        return nothing
    end

    if d == 7
        @inbounds for j in 1:length(vn.neighbours)
            cn_idx, _ = vn.neighbours[j]
            idx = vn.pos_in_check_neighbour[j]
            cn = tg.check_nodes[cn_idx]

            _, var, beta, beta1, u_d = build_lsd_input_excluding_precomputed_d7!(
                ws,
                j,
                total_inv_var,
                total_mean_invvar,
                channel_inv_var,
                channel_mean_invvar,
                channel_t_raw,
                channel_g,
                channel_p,
                channel_coeff,
                channel_beta_contrib,
            )
            count = simplified_lsd_candidates_d7!(ws, var, beta, beta1, u_d)
            if count > 0
                moment_match_candidates!(cn.messages[idx], ws, count, var)
            end
        end
        return nothing
    end

    @inbounds for j in 1:length(vn.neighbours)
        cn_idx, _ = vn.neighbours[j]
        idx = vn.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]

        n, var, beta, beta1, u_d = build_lsd_input_excluding_precomputed!(
            ws,
            d,
            j,
            total_inv_var,
            total_mean_invvar,
            channel_inv_var,
            channel_mean_invvar,
            channel_t_raw,
            channel_g,
            channel_p,
            channel_coeff,
            channel_beta_contrib,
        )
        count = simplified_lsd_candidates!(ws, n, var, beta, beta1, u_d)
        if count > 0
            moment_match_candidates!(cn.messages[idx], ws, count, var)
        end
    end
    return nothing
end

function lsd_variable_node_decision_optimized!(bp_result::Vector{Float64}, tg::LD.TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    d = length(vn.messages)
    n = d + 1
    ws = get_lsd_workspace(n)

    if n == 2
        @inbounds begin
            msg1 = vn.messages[1]
            if isapprox(msg1.var, LD.MIN_VAR)
                bp_result[vn_idx] = msg1.mean
                return 1
            end
            msg2 = vn.message
            if isapprox(msg2.var, LD.MIN_VAR)
                bp_result[vn_idx] = msg2.mean
                return 1
            end
        end
    end

    n, var, beta, beta1, u_d = build_lsd_input_all!(ws, vn)
    count = simplified_lsd_candidates!(ws, n, var, beta, beta1, u_d)
    if count > 0
        moment_match_candidates!(vn.message, ws, count, var)
    end

    bp_result[vn_idx] = vn.message.mean
    return count
end

lsd_variable_node_decision_optimized!(tg::LD.TannerGraph, vn_idx::Int64) =
    lsd_variable_node_decision_optimized!(tg.bp_result, tg, vn_idx)
