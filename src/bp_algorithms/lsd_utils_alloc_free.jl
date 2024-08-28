const EPSILON = 1e-30
const MAX_ITER = 10_000

mutable struct ListSphereDecodingMemory
    # NEED INPUT:
    Var::Float64
    β::Float64
    β1::Float64
    u_d::Float64
    node_degree::Int
    found_messages::Int

    # length d
    f_vector::Vector{Float64}
    g_vector::Vector{Float64}
    p_vector::Vector{Float64}
    t_vector::Vector{Float64}
    R_vector::Vector{Float64}


    # NEED simplified_lsd
    # length d
    dist_vector::Vector{Float64}
    z_vector::Vector{Float64}
    s_vector::Vector{Float64}
    gamma_vector::Vector{Float64}
    u_vector::Vector{Float64}
    msg_vector::Vector{gaussian_log_weight}
    L::Vector{Vector{Int16}}  # What is the max length?
    D::Vector{Float64}  # What is the max length?
    candidate_gaussians::Vector{gaussian_log_weight}
    moment_matching_ws::Vector{Float64}
end

function init_lsd_memory(node_degree::Int64; max_messages::Int=32)

    return ListSphereDecodingMemory(
        0.0,        # Var::Float64
        0.0,        # β::Float64
        0.0,        # β1::Float64
        0.0,        # u_d::Float64
        node_degree,        # node_degree::Int
        0,  # found messages
        Vector{Float64}(undef, node_degree),            # f_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # g_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # p_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # t_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # R_vector::Vector{Float64}
        zeros(Float64, node_degree), # dist_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # z_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # s_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # gamma_vector::Vector{Float64}
        Vector{Float64}(undef, node_degree), # u_vector::Vector{Float64}
        Vector{gaussian_log_weight}([gaussian_log_weight(0.0, 0.0) for k = 1:node_degree]),# msg_vector::Vector{gaussian_log_weight}
        Vector{Vector{Int16}}([Vector{Int16}(undef, node_degree) for k = 1:max_messages]),
        Vector{Float64}(undef, max_messages),
        Vector{gaussian_log_weight}([gaussian_log_weight(0.0, 0.0) for k = 1:max_messages]),
        Vector{Float64}(undef, max_messages)
    )
end

function ListSphereDecodingInput!(mem::ListSphereDecodingMemory)

    Vinv = 0.0
    β = 2.0

    for i = 1:mem.node_degree
        msg = mem.msg_vector[i]
        Vinv += 1 / msg.var
        mem.t_vector[i] = sign(msg.period) / sqrt(msg.var)
        mem.g_vector[i] = sqrt(msg.var * msg.period^2)
        mem.p_vector[i] = msg.mean * msg.period
        # #println("Var & Period ", msg.var, " ", msg.period, " ")
        β = abs(msg.period) < W_MIN ? max(β, 1 / sqrt(msg.var * msg.period^2)) : β  # Wang & Mow: Eq. (44)
        # β = max(β, 1 / sqrt(msg.var * msg.period^2))
    end

    mem.u_d = mem.msg_vector[end].mean / mem.msg_vector[end].var / sqrt(Vinv)
    mem.t_vector .*= 1 / sqrt(Vinv)
    # println(mem.t_vector)
    update_f_vector!(mem)
    update_R_vector!(mem)

    mem.β = β
    mem.β1 = β
    mem.Var = 1 / Vinv

    return
end

function update_f_vector!(mem::ListSphereDecodingMemory)
    mem.f_vector[1] = 1 - mem.t_vector[1]^2
    for i = 2:mem.node_degree
        mem.f_vector[i] = mem.f_vector[i-1] - mem.t_vector[i]^2
    end
    return
end

function update_R_vector!(mem::ListSphereDecodingMemory)
    mem.R_vector[1] = 1 / mem.g_vector[1]^2 * mem.f_vector[1]
    for i = 2:mem.node_degree
        mem.R_vector[i] = 1 / mem.g_vector[i]^2 * mem.f_vector[i] / mem.f_vector[i-1]
    end
    return
end

function update_msg_vector!(mem::ListSphereDecodingMemory, vn::VariableNode, j::Int64)
    cnt = 1
    for i = 1:mem.node_degree
        if i != j
            mem.msg_vector[cnt] = vn.messages[i]
            cnt += 1
        end
    end
    mem.msg_vector[cnt] = vn.message
    return
end


function update_message_vector!(mem::ListSphereDecodingMemory, vn::VariableNode)
    cnt = 1
    n_neighbours = length(vn.neighbours)
    for i = 1:n_neighbours
        mem.msg_vector[cnt] = vn.messages[i]
        cnt += 1
    end
    mem.msg_vector[cnt] = vn.message
    return
end

function simplified_lsd!(mem::ListSphereDecodingMemory)
    # create pointers
    #println("Starting simplified_lsd! - line 1")
    p = mem.p_vector
    t = mem.t_vector
    g = mem.g_vector
    f = mem.f_vector
    R_sq = mem.R_vector
    dist = mem.dist_vector
    z = mem.z_vector
    s = mem.s_vector
    gamma = mem.gamma_vector
    u = mem.u_vector

    L = mem.L
    D = mem.D

    d = mem.node_degree
    k = d - 1
    u[d] = mem.u_d

    # Steps 2-5:
    gamma[k] = -p[k] + t[k] * g[k] * u[k+1] / f[k]
    z[k] = round(gamma[k])
    s[k] = sign(gamma[k] - z[k])
    dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
    iter = 0
    mem.found_messages = 0

    #println("mem.found_messages: ", mem.found_messages)
    #println("mem.dist: ", dist)
    #println("mem.β: ", mem.β)


    # println("Starting simplified_lsd! with ", mem.found_messages)
    while k <= (d - 1) && iter <= MAX_ITER && mem.found_messages < length(L)
        iter += 1
        if dist[k] <= (mem.β)^2
            if k == 1
                mem.found_messages += 1
                #println("Found messages: ", mem.found_messages)
                L[mem.found_messages] .= round.(Int16, z)
                @views D[mem.found_messages] = copy(dist[k])
                if mem.found_messages == 1
                    update_beta!(mem, D[1])
                end


                # Steps 10-12
                z[k] += s[k]
                s[k] = -s[k] - sign(s[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]

            else
                u[k] = t[k] * (z[k] + p[k]) / g[k] + u[k+1]
                k -= 1

                # Repeat Steps 2-5
                gamma[k] = -p[k] + t[k] * g[k] * u[k+1] / f[k]
                z[k] = round(gamma[k])
                s[k] = sign(gamma[k] - z[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
            end # if k == 1

        else
            if k == (d - 1)
                # println("Returned found messages: ", mem.found_messages)
                # printstyled("Returnd after $(iter) iterations.\n", color=:blue)
                return
            else
                k += 1

                # Repeat Steps 10-12
                z[k] += s[k]
                s[k] = -s[k] - sign(s[k])
                dist[k] = dist[k+1] + (gamma[k] - z[k])^2 * R_sq[k]
            end # if k == (d-1)

        end # if dist
    end # while
    # println("Returned found messages: ", mem.found_messages)
    return
end

function update_candidate_gaussians!(mem::ListSphereDecodingMemory)
    # println("Updating candidate gaussians!")
    # println("Found messages: ", mem.found_messages)
    for i = 1:mem.found_messages
        mean = 0.0
        log_weight = -1 / 2 * mem.D[i]
        for j = 1:mem.node_degree
            msg = mem.msg_vector[j]
            mean += (msg.mean + mem.L[i][j] / msg.period) / msg.var
        end
        mean *= mem.Var

        mem.candidate_gaussians[i].mean = 1.0 * mean
        mem.candidate_gaussians[i].var = 1.0 * mem.Var
        mem.candidate_gaussians[i].log_weight = 1.0 * log_weight
    end
    return
end


"""
    update_beta!(inputs::ListSphereDecodingInput, DB::Float64, ϵ=1e-10::Float64)

Updates the β parameter of the List Sphere Decoding algorithm, see Wang & Mow: Eq. (45).
"""
function update_beta!(mem::ListSphereDecodingMemory, DB::Float64, ϵ=EPSILON::Float64)
    mem.β = min(mem.β1, sqrt(DB - 2 * log(ϵ)))
end



"""
    create_lsd_memory(H::SparseMatrixCSC)

Check all distinct columns weights in the matrix `H` and initialize a `ListSphereDecodingMemory` structure
for each one in a dict with the respective key being the column weight.
"""
function create_lsd_memory(H::SparseMatrixCSC)

    mem_dict = Dict{Int64,ListSphereDecodingMemory}()


    # iteration dicts
    for i in 1:size(H, 2)
        col = H[:, i]
        weight = length(col.nzval)
        if weight in keys(mem_dict)
            continue
        else
            # #println("Initializing memory for weight: ", weight)
            mem_dict[weight] = init_lsd_memory(weight, max_messages=32)
        end
    end


    # decision dicts
    for i in 1:size(H, 2)
        col = H[:, i]
        weight = length(col.nzval) + 1
        if weight in keys(mem_dict)
            continue
        else
            # #println("Initializing memory for weight: ", weight)
            mem_dict[weight] = init_lsd_memory(weight, max_messages=32)
        end
    end


    return mem_dict
end

