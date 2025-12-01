# include("gaussians_log_weight.jl")
include("tanner_graph_log_weight.jl")


mutable struct FBAlloc
    forward::Vector{Vector{gaussian_log_weight}}
    backward::Vector{Vector{gaussian_log_weight}}
    temp_mix::Vector{gaussian_log_weight}
    outputs::Vector{gaussian_log_weight}
end

function FBAlloc(d::Int, max_mix::Int)
    forward = [[gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ = 1:max_mix] for _ in 1:d]
    backward = [[gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ = 1:max_mix] for _ in 1:d]
    temp_mix = [gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ in 1:max_mix]
    outputs = [gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ in 1:d]
    return FBAlloc(forward, backward, temp_mix, outputs)
end

mutable struct LDLCDecoder
    tg::TannerGraph
    fb_allocs::Vector{FBAlloc}  # one allocation per variable node
    M::Int                      # mixture size
end

"""
    LDLCDecoder(tg, M)

Initialize LDLC decoder with preallocated buffers for forward-backward recursion.
"""
function LDLCDecoder(tg::TannerGraph, M::Int)
    fb_allocs = Vector{FBAlloc}(undef, tg.nv)
    for vn_idx in 1:tg.nv
        d = length(tg.var_nodes[vn_idx].neighbours)
        fb_allocs[vn_idx] = FBAlloc(d, M)
    end
    return LDLCDecoder(tg, fb_allocs, M)
end


function forward_backward_decoder!(alloc::FBAlloc, R_sets::Vector{Vector{gaussian_log_weight}}, channel::gaussian_log_weight)
    d = length(R_sets)

    # Forward recursion
    alloc.forward[1] .= R_sets[1]
    for i in 2:d
        combine_mixture!(alloc.forward[i], alloc.forward[i-1], R_sets[i], alloc.temp_mix)
    end

    # Backward recursion
    alloc.backward[d] .= R_sets[d]
    for i in (d-1):-1:1
        combine_mixture!(alloc.backward[i], alloc.backward[i+1], R_sets[i], alloc.temp_mix)
    end

    # Compute outputs
    for i in 1:d
        left = (i > 1) ? alloc.forward[i-1] : []
        right = (i < d) ? alloc.backward[i+1] : []
        combine_mixture!(alloc.temp_mix, left, right, alloc.temp_mix)
        combine_mixture!(alloc.temp_mix, alloc.temp_mix, [channel], alloc.temp_mix)
        moment_matching!(alloc.outputs[i], alloc.temp_mix)
    end

    return alloc.outputs
end



function decision_step!(dec::LDLCDecoder)
    tg = dec.tg
    for vn_idx in 1:tg.nv
        vn = tg.var_nodes[vn_idx]
        combined = vn.message
        for msg in vn.messages
            prod!(combined, msg)
        end
        tg.bp_result[vn_idx] = combined.mean
    end
end



function combine_mixture!(dest::Vector{gaussian_log_weight}, mix1::Vector{gaussian_log_weight}, mix2::Vector{gaussian_log_weight}, temp::Vector{gaussian_log_weight})
    if isempty(mix1)
        dest .= mix2
        return
    elseif isempty(mix2)
        dest .= mix1
        return
    end
    idx = 1
    for g1 in mix1
        for g2 in mix2
            temp[idx].mean = (1 / (1 / g1.var + 1 / g2.var)) * (g1.mean / g1.var + g2.mean / g2.var)
            temp[idx].var = max(1 / (1 / g1.var + 1 / g2.var), MIN_VAR)
            temp[idx].log_weight = -(g1.mean - g2.mean)^2 / (2 * (g1.var + g2.var)) - log(sqrt(2 * pi * (g1.var + g2.var))) + g1.log_weight + g2.log_weight
            idx += 1
        end
    end
    resize!(dest, idx - 1)
    dest .= temp[1:idx-1]
end


H = [
    0    -0.8   0    -0.5   1     0;
    0.8   0     0     1     0    -0.5;
    0     0.5   1     0     0.8   0;
    0     0    -0.5  -0.8   0     1;
    1     0     0     0     0.5   0.8;
    0.5  -1    -0.8   0     0     0
]


tg = initialize_tanner_graph(H)
M = 2

decoder = LDLCDecoder(tg, M)

function run_decoder!(dec::LDLCDecoder, message:Vector{Float64}, σ::Float64, max_iter::Int64)


    initialize_messages!(dec.tg, message, σ)

    for i = 1:max_iter
        check_node_iterations!(decoder)
        variable_node_iterations!(decoder)
    end

    decision_step!(decoder)

    return dec.tg.bp_result

end

