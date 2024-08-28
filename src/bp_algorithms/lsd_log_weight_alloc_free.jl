function lsd_variable_node_message!(cn_message::gaussian_log_weight, vn::VariableNode, nb_idx::Int, mem::ListSphereDecodingMemory)
    update_msg_vector!(mem, vn, nb_idx)
    # println("Starting simplified_lsd!")
    ListSphereDecodingInput!(mem)
    simplified_lsd!(mem)
    update_candidate_gaussians!(mem)
    if mem.found_messages > 0
        moment_matching!(cn_message, mem.candidate_gaussians[1:mem.found_messages])
    end
    return
end


function lsd_variable_node_messages_alloc_free!(tg::TannerGraph, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    n_neighbours = length(vn.neighbours)
    # println("lsd_alloc_free")
    for j = 1:n_neighbours
        cn_idx, _ = vn.neighbours[j]
        idx = vn.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]
        lsd_variable_node_message!(cn.messages[idx], vn, j, tg.lsd_mem[n_neighbours])
    end
    return
end


"""
    _lsd_variable_node_decision!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)

Performs the decision step using the List Sphere Decoding algorithm for a variable node `vn_idx` in the Tanner graph `tg`.
"""
function _lsd_variable_node_decision_alloc_free!(bp_result::Vector{Float64}, tg::TannerGraph, vn_idx::Int64)
    # println("LSD Variable Node Decision")
    vn = tg.var_nodes[vn_idx]
    n_neighbours = length(vn.neighbours) + 1
    mem = tg.lsd_mem[n_neighbours]
    update_message_vector!(mem, vn)
    ListSphereDecodingInput!(mem)
    simplified_lsd!(mem)
    update_candidate_gaussians!(mem)
    if mem.found_messages > 0
        moment_matching!(vn.message, mem.candidate_gaussians[1:mem.found_messages])
    end
    bp_result[vn_idx] = vn.message.mean
    return
end
