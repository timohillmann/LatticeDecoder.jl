using FFTW
using SparseArrays
using LinearAlgebra
using Plots
using Statistics
FFTW.set_num_threads(Sys.CPU_THREADS)

# -------------------------
# Message & node types
# -------------------------
mutable struct QuantizedMessage
    pdf::Vector{Float64}   # length L
    Δ::Float64
end

mutable struct CheckNodeQuant
    neighbours::Vector{Tuple{Int64,Float64}}   # (var_index, h_ij)
    messages::Vector{QuantizedMessage}         # C->V messages
    pos_in_var_neighbour::Vector{Int64}
end

mutable struct VariableNodeQuant
    id::Int64
    neighbours::Vector{Tuple{Int64,Float64}}   # (check_index, h_ij)
    message::QuantizedMessage                   # channel prior
    messages::Vector{QuantizedMessage}         # outgoing V->C messages
    pos_in_check_neighbour::Vector{Int64}
end

mutable struct TannerGraphQuant
    var_nodes::Vector{VariableNodeQuant}
    check_nodes::Vector{CheckNodeQuant}
    var_node_to_posit::Dict{Int64,Int64}
    nv::Int64
    nc::Int64

    # quantization params
    L::Int
    Δ::Float64
    grid::Vector{Float64}

    # runtime
    bp_result::Vector{Float64}
end

function initialize_tanner_graph_quant(H::SparseMatrixCSC; L::Int=1024, Δ::Float64=1/256)
    nc, nv = size(H)
    var_nodes = Vector{VariableNodeQuant}(undef, nv)
    check_nodes = Vector{CheckNodeQuant}(undef, nc)
    var_node_to_posit = Dict{Int64,Int64}()

    node_to_stab = Dict{Int64, Vector{Tuple{Int64,Float64}}}()
    for c = 1:nc
        nzind, nzval = H[c,:].nzind, H[c,:].nzval
        neighbours = [(nzind[k], nzval[k]) for k = 1:length(nzind)]
        pos_in_var_neighbour = Int64[]
        for k = 1:length(neighbours)
            v,h = neighbours[k]
            if haskey(node_to_stab,v)
                push!(node_to_stab[v], (c,h))
                push!(pos_in_var_neighbour, length(node_to_stab[v]))
            else
                node_to_stab[v] = [(c,h)]
                push!(pos_in_var_neighbour, 1)
            end
        end
        check_nodes[c] = CheckNodeQuant(neighbours, [QuantizedMessage(zeros(Float64,L), Δ) for _ in neighbours], pos_in_var_neighbour)
    end

    sorted = sort(collect(node_to_stab), by=x->x[1])
    counter = 1
    for (v, neighs) in sorted
        var_node_to_posit[v] = counter
        ch_msg = QuantizedMessage(zeros(Float64,L), Δ)
        msgs = [QuantizedMessage(zeros(Float64,L), Δ) for _ in neighs]
        var_nodes[counter] = VariableNodeQuant(v, neighs, ch_msg, msgs, Int64[])
        counter += 1
    end

    for c = 1:nc
        row = H[c,:]
        for j = 1:length(row.nzind)
            v = row.nzind[j]
            vp = var_node_to_posit[v]
            push!(var_nodes[vp].pos_in_check_neighbour, j)
        end
    end

    grid = make_grid(Δ,L)
    bp_result = zeros(Float64,nv)
    #return TannerGraphQuant(var_nodes, check_nodes, var_node_to_posit, nv, nc, L, Δ, grid, bp_result)
    return TannerGraphQuant(var_nodes, check_nodes, var_node_to_posit, nv, nc, L, Δ, grid, bp_result)
end

initialize_tanner_graph_quant(H::Matrix, L::Int=1024, Δ::Float64=1/256) = initialize_tanner_graph_quant(sparse(H); L=L, Δ=Δ)


# -------------------------
# Utilities
# -------------------------
make_grid(Δ::Float64, L::Int) = collect(-L*Δ/2 : Δ : (L*Δ/2 - Δ));

function normalize!(msg::QuantizedMessage)
    norm = sum(msg.pdf * msg.Δ)
    msg.pdf /= norm
    return nothing
end

function scale_msg(msg::QuantizedMessage, h::Float64; lw::Int=1)
    L = length(msg.pdf)

    # trivial case: h ≈ 1 → no expansion
    if isapprox(h, 1.0; atol=1e-12)
        return msg
    end

    # allocate temporary array
    tmp = similar(msg.pdf)

    center = (L + 1) / 2
    scale = h  # stretch factor

    @inbounds for k in 1:L
        # target floating-point index in original PDF
        idxf = ((k - center) / scale) + center

        # linear interpolation
        i0 = floor(Int, idxf)
        i1 = i0 + 1
        t = idxf - i0

        # clamp indices
        i0c = clamp(i0, 1, L)
        i1c = clamp(i1, 1, L)

        tmp[k] = (1 - t) * msg.pdf[i0c] + t * msg.pdf[i1c]
    end

    # optional ±lw averaging to avoid impulses disappearing
    if lw > 0
        tmp2 = copy(tmp)
        @inbounds for k in 1:L
            il = max(1, k - lw)
            ir = min(L, k + lw)
            tmp[k] = mean(tmp2[il:ir])
        end
    end

    # write back and normalize


    out =  QuantizedMessage(tmp, msg.Δ)
    normalize!(out)
    return out
end

function scale_msg!(msg::QuantizedMessage, h::Float64; lw::Int=1)
    L = length(msg.pdf)
    ah = abs(h)

    # trivial case: h ≈ 1 → no expansion
    if isapprox(ah, 1.0; atol=1e-12)
        return nothing
    end

    # allocate temporary array
    tmp = similar(msg.pdf)

    center = (L + 1) / 2
    scale = ah  # stretch factor

    @inbounds for k in 1:L
        # target floating-point index in original PDF
        idxf = ((k - center) / scale) + center

        # linear interpolation
        i0 = floor(Int, idxf)
        i1 = i0 + 1
        t = idxf - i0

        # clamp indices
        i0c = clamp(i0, 1, L)
        i1c = clamp(i1, 1, L)

        tmp[k] = (1 - t) * msg.pdf[i0c] + t * msg.pdf[i1c]
    end

    # optional ±lw averaging to avoid impulses disappearing
    if lw > 0
        tmp2 = copy(tmp)
        @inbounds for k in 1:L
            il = max(1, k - lw)
            ir = min(L, k + lw)
            tmp[k] = mean(tmp2[il:ir])
        end
    end

    # write back and normalize
    msg.pdf .= tmp
    normalize!(msg)

    return nothing
end

function shift_pdf(pdf::Vector{Float64}, Δ::Float64, shift::Float64)
    N = length(pdf)
    shifted = zeros(Float64, N)

    for k in 1:N
        xk = (k - 1) * Δ            # sample location
        u = xk - shift             # shifted location

        # outside domain → zero
        if u < 0 || u > (N-1)*Δ
            shifted[k] = 0.0
            continue
        end

        idxf = u / Δ + 1           # fractional index
        i0 = floor(Int, idxf)
        t  = idxf - i0             # interpolation weight

        # bounds: if exactly at the boundary, clamp
        i1 = max(1, min(N, i0))
        i2 = (i1 == N) ? N : (i1 + 1)

        # linear interpolation
        shifted[k] = (1-t)*pdf[i1] + t*pdf[i2]
    end

    return shifted
end

function shift!(msg::QuantizedMessage, shift::Float64)
    msg.pdf .= shift_pdf(msg.pdf, msg.Δ, shift)
    return nothing
end


function convolve_fft(msgs::AbstractArray{QuantizedMessage})
    d = length(msgs)
    L = length(msgs[1].pdf)
    fft_vecs = [rfft(msg.pdf) for msg in msgs]
    prod = 1.0 * fft_vecs[1]
    @inbounds for j = 2:d
        prod .*= fft_vecs[j]
    end
    conv_pdf = irfft(prod, L)
    return conv_pdf    
end

function convolve_fft!(msg_out::QuantizedMessage, msgs::AbstractVector{QuantizedMessage})
    d = length(msgs)
    L = length(msgs[1].pdf)
    fft_vecs = [rfft(msg.pdf) for msg in msgs]
    prod = fft_vecs[1]
    @inbounds for j = 2:d
        prod .*= fft_vecs[j]
    end
    
    msg_out.pdf .= irfft(prod, L)
end


# Init messages

"""
    initialize_messages!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Float64)

Initialize the messages of the Tanner graph for LDLC decoding.
"""
function initialize_messages!(tg::TannerGraphQuant, syndrome::Vector{Float64}, σ::Float64)

    # Set variable node messages to received channel message
    for i in 1:length(tg.var_nodes)
        @. tg.var_nodes[i].message.pdf = exp(-0.5*((tg.grid-syndrome[i])/σ)^2) / sqrt(2*π*σ^2)
    end

    # Collect the messages from the neighbouring variable nodes
    for i in 1:length(tg.check_nodes)
        check_node = tg.check_nodes[i]

        for idx in 1:length(check_node.neighbours)
            vn_idx, edge_weight = check_node.neighbours[idx]
            @. check_node.messages[idx].pdf = 1.0 * tg.var_nodes[vn_idx].message.pdf
        end
    end
end


"""
    initialize_messages!(tg::TannerGraph, syndrome::Vector{Float64}, σ::Vector{Float64})

Initialize the messages of the Tanner graph for LDLC decoding. NOTE: here the variances can be different for different variable nodes
"""
function initialize_messages!(tg::TannerGraphQuant, syndrome::Vector{Float64}, σ::Vector{Float64})

    # Set variable node messages to received channel message
    for i in 1:length(tg.var_nodes)
        @. tg.var_nodes[i].message.pdf = exp(-0.5*((tg.grid-syndrome[i])/σ[i])^2) / sqrt(2*π*σ[i]^2)
    end

    # Collect the messages from the neighbouring variable nodes
    for i in 1:length(tg.check_nodes)
        check_node = tg.check_nodes[i]

        for idx in 1:length(check_node.neighbours)
            vn_idx, edge_weight = check_node.neighbours[idx]
            @. check_node.messages[idx].pdf = 1.0 * tg.var_nodes[vn_idx].message.pdf
        end
    end
end


# Check node message
function check_node_messages!(tg::TannerGraphQuant, cn_idx::Int64)
    check_node = tg.check_nodes[cn_idx]
    d = length(check_node.neighbours)


    # Get message vectors
    msgs_scaled = Vector{QuantizedMessage}(undef, d)
    @inbounds for i = 1:d
        msgs_scaled[i] = scale_msg(
            check_node.messages[i],
            1 / check_node.neighbours[i][2]
        )
    end

    # copmpute the new messages for the neighbouring variable nodes
    @inbounds for i = 1:d
        vn_idx, edge_weight = check_node.neighbours[i]
        idx = check_node.pos_in_var_neighbour[i]
        vn = tg.var_nodes[vn_idx]
        
        mask = setdiff(1:d, i)
        msgs = @view msgs_scaled[mask]
        update_vn_message!(vn.messages[idx], edge_weight, msgs)
    end
end

function update_vn_message!(vn_msg::QuantizedMessage, h::Float64, msgs::AbstractArray{QuantizedMessage})
    # convolve -> strech -> extend

    convolve_fft!(vn_msg, msgs)
    normalize!(vn_msg)

    scale_msg!(vn_msg, -h)
    normalize!(vn_msg)

    tmp = 1.0 * vn_msg.pdf
    for j = 1:2
        vn_msg.pdf .+= shift_pdf(tmp, vn_msg.Δ, +j/h)
        vn_msg.pdf .+= shift_pdf(tmp, vn_msg.Δ, -j/h)
    end
    
    normalize!(vn_msg)
end

"""
    check_node_iterations!(tg::TannerGraph)

    Iterates over all check nodes and updates the messages of the variable nodes.
"""
function check_node_iterations!(tg::TannerGraphQuant)
    for i in 1:tg.nc
        check_node_messages!(tg, i)
    end
end



# VariableNode messages
import Base.prod!
function prod!(msg::QuantizedMessage, msg2::QuantizedMessage)
    msg.pdf .*= msg2.pdf
end

function prod!(msg::QuantizedMessage, msgs::Vector{QuantizedMessage})
    for j = 1:length(msgs)
        msg.pdf .*= msgs[j].pdf
    end
end

function variable_node_messages!(tg::TannerGraphQuant, vn_idx::Int64)
    var_node = tg.var_nodes[vn_idx]
    
    for j = 1:length(var_node.neighbours)
        cn_idx, edge_weight = var_node.neighbours[j]
        idx = var_node.pos_in_check_neighbour[j]
        cn = tg.check_nodes[cn_idx]
        tmp = 1.0 * var_node.message.pdf
        for i = 1:length(var_node.messages)
            if i != j
                tmp .*= var_node.messages[i].pdf
            end
        end
        cn.messages[idx].pdf .= 1.0 * tmp
        normalize!(cn.messages[idx])
    end
end

function variable_node_iterations!(tg::TannerGraphQuant)
    for i = 1:tg.nv
        variable_node_messages!(tg, i)
    end
end

function variable_node_decision!(tg::TannerGraphQuant, vn_idx::Int64)
    vn = tg.var_nodes[vn_idx]
    prod!(vn.message, vn.messages)
    tg.bp_result[vn_idx] = tg.grid[argmax(vn.message.pdf)]
end

function decision_step!(tg::TannerGraphQuant)
    for i in 1:tg.nv
        variable_node_decision!(tg, i)
    end
end


function run_belief_propagation!(tg::TannerGraphQuant, message::Vector{Float64}, σ::Float64, max_iter::Int64)

    initialize_messages!(tg, message, σ)
    for i = 1:max_iter
        check_node_iterations!(tg)
        variable_node_iterations!(tg)
    end
    
    decision_step!(tg)
    return tg.bp_result
end