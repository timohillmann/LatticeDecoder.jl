# # include("gaussians_log_weight.jl")
# include("tanner_graph_log_weight.jl")

# const MIN_VAR::Float64 = 1e-10

# mutable struct FBAlloc
#     forward::Vector{Vector{gaussian_log_weight}}
#     backward::Vector{Vector{gaussian_log_weight}}
#     temp_mix::Vector{gaussian_log_weight}
#     outputs::Vector{gaussian_log_weight}
# end

# function FBAlloc(d::Int, max_mix::Int)
#     forward = [[gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ = 1:max_mix] for _ in 1:d]
#     backward = [[gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ = 1:max_mix] for _ in 1:d]
#     temp_mix = [gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ in 1:max_mix]
#     outputs = [gaussian_log_weight(0.0, 1.0, 0.0, 1.0) for _ in 1:d]
#     return FBAlloc(forward, backward, temp_mix, outputs)
# end

# mutable struct LDLCDecoder
#     tg::TannerGraph
#     fb_allocs::Vector{FBAlloc}  # one allocation per variable node
#     M::Int                      # mixture size
# end

# """
#     LDLCDecoder(tg, M)

# Initialize LDLC decoder with preallocated buffers for forward-backward recursion.
# """
# function LDLCDecoder(tg::TannerGraph, M::Int)
#     fb_allocs = Vector{FBAlloc}(undef, tg.nv)
#     for vn_idx in 1:tg.nv
#         d = length(tg.var_nodes[vn_idx].neighbours)
#         fb_allocs[vn_idx] = FBAlloc(d, M)
#     end
#     return LDLCDecoder(tg, fb_allocs, M)
# end


# function variable_node_messages_M_gaussian_allocationless!(tg::TannerGraph, vn_idx::Int64, decoder::LDLCDecoder)
#     var_node = tg.var_nodes[vn_idx]
#     d = length(var_node.neighbours)

#     @inbounds for j = 1:d
#         cn_idx, edge_weight = var_node.neighbours[j]   
#         @views m_nearest!(decoder.mixture_alloc[j], var_node.message[j], var_node.message.mean, decoder.M)
#     end

#     forward_backward_recursion!(decoder, var_node.message, d)

#     for j = 1:d
#         cn_idx, _ = var_node.neighbours[j]
#         idx = var_node.pos_in_check_neighbour[j]
#         cn = tg.check_nodes[cn_idx]

#         cn.messages[idx].mean = decoder.outputs_alloc[j].mean
#         cn.messages[idx].var = decoder.outputs_alloc[j].var
#         # cn.messages[idx].log_weight = outputs[j].log_weight
#     end

# end


# function m_nearest!(alloc::Vector{gaussian_log_weight}, g::gaussian_log_weight, y::Float64, M::Int64)
#     m = g.mean
#     rhs = (m - y) * h
#     center = round(rhs)
#     offset = M ÷ 2
#     @inbounds for k = 1:M
#         b_k = center - (k - offset - (M % 2))
#         alloc[k].vean = m - (b_k / h)
#         alloc[k].var = g.var
#         #  = gaussian_log_weight(m - (b_k / h), g.var)
#     end
#     return gs
# end


# function forward_backward_recursion!(decoder:LDLCDecoder, g::gaussian_log_weight, d::Int64)

#     forward = decoder.forward_alloc
#     backward = decoder.backward_alloc

#     forward[1] = decoder.mixture_alloc[1]
#     for j = 2:d
#         prod!(forward[j], forward[j-1], decoder.mixture_alloc[j])
#     end

#     backward = decoder.backward_alloc[d]
#     for j = (d-1):-1:1
#         prod!(backward[j], backward[j+1], decoder.mixture_alloc[j])
#     end
    
#     for j = 1:d
#         if j == 1
#             prod!(decoder.mixture_alloc[j], backward[j+1], g)
#             @views moment_matching!(decoder.outputs_alloc[j], mixtures[j])
#         elseif  j == d
#             prod!(decoder.mixture_alloc[j], forward[j-1], g)
#             @views moment_matching!(decoder.outputs_alloc[j], mixtures[j])
#         else
#             prod!(decoder.mixture_alloc[j], forward[j-1], backward[j+1])
#             prod!(decoder.mixture_alloc[j], g)
#             @views moment_matching!(decoder.outputs_alloc[j], mixtures[j])
#         end
#     end

# end



# function Base.prod!(dest::AbstractVector{gaussian_log_weight}, gs::AbstractVector{gaussian_log_weight}, g::gaussian_log_weight)
#     n1 = length(gs)
#     for idx = 1:n1
#         @views prod!(dest[idx], gs[idx], g)
#     end
#     return nothing
# end

# function Base.prod!(dest::AbstractVector{gaussian_log_weight}, gs1::AbstractVector{gaussian_log_weight}, gs2::AbstractVector{gaussian_log_weight})
#     n1 = length(gs1)
#     n2 = length(gs2)
#     idx = 1
#     for idx1 = 1:n1
#         for idx2 = 1:n2
#             @views prod!(dest[idx], gs1[idx1], gs2[idx2])
#             idx += 1
#         end
#     end
#     return nothing
# end

# function Base.prod!(g::gaussian_log_weight, g1::gaussian_log_weight, g2::gaussian_log_weight)
#     m1 = g1.mean
#     m2 = g2.mean
#     Δ1 = g1.var
#     Δ2 = g2.var

#     Δ = max(1 / (1 / Δ1 + 1 / Δ2), MIN_VAR)
#     m = Δ * (m1 / Δ1 + m2 / Δ2)
#     c = -(m1 - m2)^2 / (2 * (Δ1 + Δ2)) - log((sqrt(2 * pi * (Δ1 + Δ2))))

#     g.mean = m
#     g.var = max(Δ, MIN_VAR)
#     g.log_weight = c + g1.log_weight + g2.log_weight    
#     return nothing
# end

# using BenchmarkTools
# import LatticeDecoder: gaussian_log_weight
# g = gaussian_log_weight(0.0, 1.0)
# g1 = gaussian_log_weight(-1.0, 0.5)
# g2 = gaussian_log_weight(1.0, 0.5)

# M = 10
# gs = gaussian_log_weight[gaussian_log_weight(1.0, 1.0) for _ = 1:M^2]
# gs1 = gaussian_log_weight[gaussian_log_weight(1.0, 1.0) for _ = 1:3]
# gs2 = gaussian_log_weight[gaussian_log_weight(-1.0, 1.0) for _ = 1:3]




# function decision_step!(dec::LDLCDecoder)
#     tg = dec.tg
#     for vn_idx in 1:tg.nv
#         vn = tg.var_nodes[vn_idx]
#         combined = vn.message
#         for msg in vn.messages
#             prod!(combined, msg)
#         end
#         tg.bp_result[vn_idx] = combined.mean
#     end
# end



# function combine_mixture!(dest::Vector{gaussian_log_weight}, mix1::Vector{gaussian_log_weight}, mix2::Vector{gaussian_log_weight}, temp::Vector{gaussian_log_weight})
#     if isempty(mix1)
#         dest .= mix2
#         return
#     elseif isempty(mix2)
#         dest .= mix1
#         return
#     end
#     idx = 1
#     for g1 in mix1
#         for g2 in mix2
#             temp[idx].mean = (1 / (1 / g1.var + 1 / g2.var)) * (g1.mean / g1.var + g2.mean / g2.var)
#             temp[idx].var = max(1 / (1 / g1.var + 1 / g2.var), MIN_VAR)
#             temp[idx].log_weight = -(g1.mean - g2.mean)^2 / (2 * (g1.var + g2.var)) - log(sqrt(2 * pi * (g1.var + g2.var))) + g1.log_weight + g2.log_weight
#             idx += 1
#         end
#     end
#     resize!(dest, idx - 1)
#     dest .= temp[1:idx-1]
# end


# H = [
#     0    -0.8   0    -0.5   1     0;
#     0.8   0     0     1     0    -0.5;
#     0     0.5   1     0     0.8   0;
#     0     0    -0.5  -0.8   0     1;
#     1     0     0     0     0.5   0.8;
#     0.5  -1    -0.8   0     0     0
# ]


# tg = initialize_tanner_graph(H)
# M = 2

# decoder = LDLCDecoder(tg, M)

# function run_decoder!(dec::LDLCDecoder, message:Vector{Float64}, σ::Float64, max_iter::Int64)


#     initialize_messages!(dec.tg, message, σ)

#     for i = 1:max_iter
#         check_node_iterations!(decoder)
#         variable_node_iterations!(decoder)
#     end

#     decision_step!(decoder)

#     return dec.tg.bp_result

# end

