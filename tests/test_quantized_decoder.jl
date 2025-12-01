using SparseArrays
using LinearAlgebra
using FFTW
using Plots
using Statistics
using LatticeDecoder
FFTW.set_num_threads(Sys.CPU_THREADS)

# # -------------------------
# # Message & node types
# # -------------------------
# mutable struct QuantizedMessage
#     pdf::Vector{Float64}   # length L
#     Δ::Float64
# end

# mutable struct CheckNodeQuant
#     neighbours::Vector{Tuple{Int64,Float64}}   # (var_index, h_ij)
#     messages::Vector{QuantizedMessage}         # C->V messages
#     pos_in_var_neighbour::Vector{Int64}
# end

# mutable struct VariableNodeQuant
#     id::Int64
#     neighbours::Vector{Tuple{Int64,Float64}}   # (check_index, h_ij)
#     message::QuantizedMessage                   # channel prior
#     messages::Vector{QuantizedMessage}         # outgoing V->C messages
#     pos_in_check_neighbour::Vector{Int64}
# end

# mutable struct TannerGraphQuant
#     var_nodes::Vector{VariableNodeQuant}
#     check_nodes::Vector{CheckNodeQuant}
#     var_node_to_posit::Dict{Int64,Int64}
#     nv::Int64
#     nc::Int64

#     # quantization params
#     L::Int
#     Δ::Float64
#     N::Int64
#     D::Int64
#     grid::Vector{Float64}

#     # runtime
#     bp_result::Vector{Float64}
# end

L = 1024
Δ = 1/256
D = Int(L * Δ)
N = Int(1/Δ)

make_grid(Δ::Float64, L::Int) = collect(-L*Δ/2 : Δ : (L*Δ/2 - Δ));
grid = make_grid(Δ, L);
# -------------------------
# Utilities
# -------------------------

function normalize!(msg::LatticeDecoder.QuantizedMessage)
    norm = sum(msg.pdf * msg.Δ)
    msg.pdf /= norm
    return nothing
end

"""
    expand_pdf_avg!(msg::QuantizedMessage, h::Float64; lw::Int=1)

In-place “stretch & average” of a quantized PDF for LDLC check-node processing.

- `msg`: the input QuantizedMessage (pdf & Δ)
- `h`: the edge weight
- `lw`: half-width of averaging window (default 1)
"""
# function expand_pdf_avg!(msg::QuantizedMessage, h::Float64; lw::Int=1)
#     L = length(msg.pdf)
#     ah = copy(h)

#     # trivial case: h ≈ 1 → no expansion
#     if isapprox(ah, 1.0; atol=1e-12)
#         return nothing
#     end

#     # allocate temporary array
#     tmp = similar(msg.pdf)

#     center = (L + 1) / 2
#     scale = ah  # stretch factor

#     @inbounds for k in 1:L
#         # target floating-point index in original PDF
#         idxf = ((k - center) / scale) + center

#         # linear interpolation
#         i0 = floor(Int, idxf)
#         i1 = i0 + 1
#         t = idxf - i0

#         # clamp indices
#         i0c = clamp(i0, 1, L)
#         i1c = clamp(i1, 1, L)

#         tmp[k] = (1 - t) * msg.pdf[i0c] + t * msg.pdf[i1c]
#     end

#     # optional ±lw averaging to avoid impulses disappearing
#     if lw > 0
#         tmp2 = copy(tmp)
#         @inbounds for k in 1:L
#             il = max(1, k - lw)
#             ir = min(L, k + lw)
#             tmp[k] = mean(tmp2[il:ir])
#         end
#     end

#     # write back and normalize
#     msg.pdf .= tmp
#     normalize!(msg)

#     return nothing
# end


# """
#     stretch_pdf!(msg::QuantizedMessage, Qtilde::Vector{Float64}, h::Float64)

# Stretch a single-period PDF Q̃ to msg.pdf in-place according to
# Q_j(x) = Q̃(-h x), using linear interpolation and periodicity.
# """
# function stretch_pdf!(msg::QuantizedMessage, Qtilde::Vector{Float64}, h::Float64)
#     L = length(msg.pdf)
#     N = length(Qtilde)
#     Δ = msg.Δ
#     center = (L + 1) / 2

#     tmp = similar(msg.pdf)

#     @inbounds for k in 1:L
#         # x-coordinate on the grid
#         x = (k - center) * Δ
#         # corresponding point in Q̃
#         u = -h * x
#         u = mod(u, 1.0)  # wrap into [0,1)

#         # linear interpolation
#         idx = u * N + 1
#         i0 = floor(Int, idx)
#         t = idx - i0

#         i1 = i0 <= 0 ? i0 + N : (i0 > N ? i0 - N : i0)
#         i2 = i1 == N ? 1 : i1 + 1

#         tmp[k] = (1 - t) * Qtilde[i1] + t * Qtilde[i2]
#     end

#     msg.pdf .= tmp
#     normalize!(msg)
#     return nothing
# end


function initialize_message!(msg::LatticeDecoder.QuantizedMessage, yk::Float64, σ::Float64)
        
    grid  = make_grid(Δ, L)

    pdf = [exp(-0.5*((x-yk)/σ)^2) for x in grid]
    msg.pdf = copy(pdf)
    normalize!(msg)
    return nothing
end 


# function decimated_fft(msg::QuantizedMessage, tg::TannerGraphQuant)
#     g = zeros(Float64, tg.N)
#     @inbounds for i = 1:tg.N
#         s = 0.0
#         @inbounds for k = 0:(tg.D-1)
#             s += msg.pdf[i + k * tg.N]
#         end
#         g[i] = s
#     end

#     return rfft(g)
# end

# function decimated_fft(msg::QuantizedMessage, D, N)
#     g = zeros(Float64, N)
#     @inbounds for i = 1:N
#         s = 0.0
#         @inbounds for k = 0:(D-1)
#             s += msg.pdf[i + k * N]
#         end
#         g[i] = s
#     end

#     return rfft(g)
# end


# function convolve_fft(msgs::Vector{QuantizedMessage})
#     d = length(msgs)
#     L = length(msgs[1].pdf)
#     fft_vecs = [rfft(msg.pdf) for msg in msgs]
#     prod = fft_vecs[1]
#     @inbounds for j = 2:d
#         prod .*= fft_vecs[j]
#     end
#     conv_pdf = irfft(prod, L)
#     return conv_pdf    
# end



# """
#     periodic_extend(Q::Vector{Float64}, h::Float64, Δ::Float64, L::Int)

# Optional: widen by adding shifted copies to mitigate narrow impulses.

# Qw[k] = Q[k-1] + Q[k] + Q[k+1]
# """
# function periodic_extend(Q::Vector{Float64})
#     L = length(Q)
#     tmp = copy(Q)
#     for k in 1:L
#         left = k > 1 ? tmp[k-1] : 0.0
#         right = k < L ? tmp[k+1] : 0.0
#         Q[k] = left + tmp[k] + right
#     end
#     normalize!(QuantizedMessage(Q, 1.0))  # Δ will be handled outside
#     return Q
# end

# function convolve_fft_sommer(msgs::Vector{QuantizedMessage}, tg::TannerGraphQuant)
#     d = length(msgs)

#     fft_vecs = [decimated_fft(msg, tg) for msg in msgs]
#     prod = fft_vecs[1]
#     @inbounds for j = 2:d
#         prod .*= fft_vecs[j]
#     end
#     conv_pdf = irfft(prod, tg.N)
#     # conv_pdf ./= sum(conv_pdf * tg.Δ)
#     return conv_pdf
# end

# function convolve_fft_sommer(msgs::Vector{QuantizedMessage}, D::Int64, N::Int64)
#     d = length(msgs)

#     fft_vecs = [decimated_fft(msg, D, N) for msg in msgs]
#     prod = fft_vecs[1]
#     @inbounds for j = 2:d
#         prod .*= fft_vecs[j]
#     end
#     conv_pdf = irfft(prod, N)
#     # conv_pdf ./= sum(conv_pdf * tg.Δ)
#     return conv_pdf
# end


# function widen_message!(pdf::Vector{Float64})
#     L = length(pdf)
#     tmp = copy(pdf)
#     @inbounds for k in 1:L
#         left  = k > 1 ? tmp[k-1] : 0.0
#         right = k < L ? tmp[k+1] : 0.0
#         pdf[k] = left + tmp[k] + right
#     end
# end

# function stretch_periodic(Q̃::Vector{Float64}, h::Float64, Δ::Float64, L::Int)
#     N = length(Q̃)
#     center = (L+1)/2
#     out = zeros(Float64, L)
#     @inbounds for k in 1:L
#         x = (k - center) * Δ
#         u = -h * x
#         u = mod(u, 1.0)          # wrap into [0,1)
#         idx = u * N + 1          # 1-based index
#         i0 = floor(Int, idx)
#         t  = idx - i0
#         i1 = i0 <= 0 ? i0 + N : i0 > N ? i0 - N : i0
#         i2 = i1 == N ? 1 : i1 + 1
#         out[k] = (1-t)*Q̃[i1] + t*Q̃[i2]
#     end
#     return out
# end

# function shift_pdf(pdf::Vector{Float64}, Δ::Float64, shift::Float64)
#     N = length(pdf)
#     shifted = zeros(Float64, N)

#     for k in 1:N
#         xk = (k - 1) * Δ            # sample location
#         u = xk - shift             # shifted location

#         # outside domain → zero
#         if u < 0 || u > (N-1)*Δ
#             shifted[k] = 0.0
#             continue
#         end

#         idxf = u / Δ + 1           # fractional index
#         i0 = floor(Int, idxf)
#         t  = idxf - i0             # interpolation weight

#         # bounds: if exactly at the boundary, clamp
#         i1 = max(1, min(N, i0))
#         i2 = (i1 == N) ? N : (i1 + 1)

#         # linear interpolation
#         shifted[k] = (1-t)*pdf[i1] + t*pdf[i2]
#     end

#     return shifted
# end


# msg = LatticeDecoder.QuantizedMessage(zeros(Float64, L), Δ);
# initialize_message!(msg, -1.0, 0.15);
# h = 1/sqrt(3)
# hs = [1.0, 1/sqrt(3), -1/sqrt(3)]
# msgs = [LatticeDecoder.QuantizedMessage(zeros(Float64, L), Δ), LatticeDecoder.QuantizedMessage(zeros(Float64, L), Δ), LatticeDecoder.QuantizedMessage(zeros(Float64, L), Δ)];
# y = [0.1, 0.0, -0.1]
# for k = 1:3
#     initialize_message!(msgs[k], y[k], 0.2)
# end

# d = 3
# msgs_scaled = Vector{LatticeDecoder.QuantizedMessage}(undef, d)
# @inbounds for i = 1:d
#     msgs_scaled[i] = LatticeDecoder.scale_msg(
#         msgs[i],
#         1 / hs[i]
#     )
# end



# LatticeDecoder.convolve_fft!(msg, msgs_scaled)
# normalize!(msg)

# LatticeDecoder.scale_msg!(msg, -h)
# normalize!(msg)

# plot(grid, [msg.pdf, tmp])

# tmp = 1.0  * msg.pdf
# msg.pdf .+= LatticeDecoder.shift_pdf(tmp, msg.Δ, +1/h)
# msg.pdf .+= LatticeDecoder.shift_pdf(tmp, msg.Δ, -1/h)
# msg.pdf .+= LatticeDecoder.shift_pdf(tmp, msg.Δ, +2/h)
# msg.pdf .+= LatticeDecoder.shift_pdf(tmp, msg.Δ, -2/h)

# normalize!(msg)



# p = plot()
# for msg in msgs
#     plot!(p, grid, msg.pdf)
# end
# display(p)

# h = 1/sqrt(3)
# for i = 1:3
#     expand_pdf_avg!(msgs[i], 1/hs[i])
# end

# Q_sommer = convolve_fft_sommer(msgs, D, N)
# Q = convolve_fft(msgs)

# Q_msg = QuantizedMessage(Q, Δ)
# normalize!(Q_msg)

# expand_pdf_avg!(Q_msg, -h)
# pdf0 = Q_msg.pdf
# pdf1 = shift_pdf(Q_msg.pdf, Δ, -2/h)
# pdf2 = shift_pdf(Q_msg.pdf, Δ, +2/h)


# pdf = pdf1 .+ pdf2 .+ pdf0

# # pdf = stretch_and_periodic(Q, h, Δ, L)
# # Qj = QuantizedMessage(Q_sommer, Δ)
# # Qj = QuantizedMessage(repeat(Q_sommer, 4), Δ)
# # expand_pdf_avg!(Qj, -h)




# # x = k * Δ
# # x - 1 / h = k * Δ -  floor( 1 / h / Δ)
# # x + 1 / h = k * Δ +  floor( 1 / h / Δ)


# # expand_pdf_avg!(msgs[1], 1/sqrt(3))

# expand_pdf_avg!(msgs[3], 1/sqrt(3))
# expand_pdf_avg!(msgs[2], 1/sqrt(3))
# expand_pdf_avg!(msgs[2], sqrt(3))

# p = plot(grid, msgs[1].pdf)
# plot!(p, grid, msgs[2].pdf)
# plot!(p, grid, msgs[3].pdf)
# display(p)


x = 1
H = [
    0    -0.8   0    -0.5   1     0;
    0.8   0     0     1     0    -0.5;
    0     0.5   1     0     0.8   0;
    0     0    -0.5  -0.8   0     1;
    1     0     0     0     0.5   0.8;
    0.5  -1    -0.8   0     0     0
]


tg = initialize_tanner_graph_quant(sparse(H); L=L, Δ=Δ);
y = 0.001 .* randn(6);
σ = 0.1;


LatticeDecoder.initialize_messages!(tg, y, σ)


# plot the first check node messages
p = plot()
for k in [3]
    for j in 1:length(tg.check_nodes[k].messages)
        plot!(p, tg.grid, tg.check_nodes[k].messages[j].pdf, label="C$k→V$j message ")
        display(p)
    end
end
vline!(p, y)



cn_idx = 1
##### CHECK NODES MESSAGES
check_node = tg.check_nodes[cn_idx];
d = length(check_node.neighbours);


# Get message vectors
msgs_scaled = Vector{LatticeDecoder.QuantizedMessage}(undef, d);
@inbounds for i = 1:d
    msgs_scaled[i] = LatticeDecoder.scale_msg(
        check_node.messages[i],
        1 / check_node.neighbours[i][2]
    )
end

# # copmpute the new messages for the neighbouring variable nodes
# @inbounds for i = [1]
#     vn_idx, edge_weight = check_node.neighbours[i]
#     idx = check_node.pos_in_var_neighbour[i]
#     vn = tg.var_nodes[vn_idx]
    
#     mask = setdiff(1:d, i)
#     msgs = @view msgs_scaled[mask]
#     LatticeDecoder.update_vn_message!(vn.messages[idx], edge_weight, msgs)
# end

i = 1

vn_idx, edge_weight = check_node.neighbours[i]
idx = check_node.pos_in_var_neighbour[i]
vn = tg.var_nodes[vn_idx]
msg = vn.messages[idx]

mask = setdiff(1:d, i)
msgs = @view msgs_scaled[mask]

msgs = msgs_scaled


# LatticeDecoder.update_vn_message!(vn.messages[idx], edge_weight, msgs)
# #### BEGIN UPDATE VN MESSAGES
# LatticeDecoder.convolve_fft!(msg, msgs);

### CONVOLVE FFT
d = length(msgs)
L = length(msgs[1].pdf)

swap_indices = vcat(collect(L÷2:L), collect(1:(L÷2-1)))

fft_vecs = [rfft(msg.pdf) for msg in msgs]


prod = 1.0 * fft_vecs[1]
@inbounds for j = 2:d
    prod .*= fft_vecs[j]
end
conv_pdf = irfft(prod, L)



LatticeDecoder.normalize!(msg);

LatticeDecoder.scale_msg!(msg, -edge_weight)
LatticeDecoder.normalize!(msg)

tmp = 1.0 * msg.pdf
for j = 1:2
    msg.pdf .+= LatticeDecoder.shift_pdf(tmp, msg.Δ, -j/edge_weight)
    msg.pdf .+= LatticeDecoder.shift_pdf(tmp, msg.Δ, +j/edge_weight)
end

normalize!(msg)
# #### END UPDATE VN MESSAGES

# plot(grid, vn.messages[idx].pdf)

p = plot()
for k in [1]
    for j in 1:length(tg.check_nodes[k].neighbours)
        vn_idx, ew = tg.check_nodes[k].neighbours[j]
        idx =  tg.check_nodes[k].pos_in_var_neighbour[j]
        plot!(p, tg.grid, tg.var_nodes[vn_idx].messages[idx].pdf, label="C$k→V$j message ")
        display(p)
    end
end
vline!(p, y)

# tg.check_nodes[1].messages[1].pdf

msgs = msgs_scaled[2:end]

d = length(msgs)

D = Int(L * Δ   )
N = L ÷ D

fft_decimated = Vector{Vector{ComplexF64}}(undef, d)
@inbounds for i = 1:d
    f = msgs[i].pdf
    g = [sum(f[j:D:end]) for j = 1:N]
    fft_decimated[i] = fft(g)  # can use fft since N is small
end

# Step 2: multiply in frequency domain
prod = fft_decimated[1]
for i = 2:d
    prod .*= fft_decimated[i]
end

# Step 3: IFFT → one period of Q̃_j(x)
Q_period = real.(ifft(prod))   # length = N_period
Q_period ./= sum(Q_period)     # normalize if needed





convolve_periodic_fft!(msg, msgs_scaled)

"""
    convolve_periodic_fft!(msg_out, msgs)

Compute convolution of messages using improved periodic FFT algorithm from Appendix VIII.
"""
function convolve_periodic_fft!(msg_out::LatticeDecoder.QuantizedMessage, msgs::Vector{LatticeDecoder.QuantizedMessage})
    Δ = msgs[1].Δ
    period_len = Int(round(1 / Δ))  # samples per period (assuming Δ divides 1)
    d = length(msgs)

    # Decimate each message to one period
    fft_vecs = Vector{Vector{ComplexF64}}(undef, d)
    for i in 1:d
        pdf = msgs[i].pdf
        L = length(pdf)
        # Group samples into bins of size L / period_len
        step = div(L, period_len)
        decimated = [sum(pdf[k:k+step-1]) for k in 1:step:L]
        fft_vecs[i] = fft(decimated)
    end

    # Multiply FFTs
    prod_fft = fft_vecs[1]
    for i in 2:d
        prod_fft .*= fft_vecs[i]
    end

    # Inverse FFT to get one period
    conv_period = real(ifft(prod_fft))

    # Stretch to full grid length
    stretched = interpolate(conv_period, period_len, length(msg_out.pdf))
    msg_out.pdf .= stretched
    normalize!(msg_out)
end



function convolve_fft!(msg_out::LatticeDecoder.QuantizedMessage, msgs::Vector{LatticeDecoder.QuantizedMessage})
    L = length(msgs[1].pdf)
    pad_len = 2L  # double length to avoid wrap-around
    fft_vecs = [rfft(vcat(msg.pdf, zeros(pad_len - L))) for msg in msgs]
    prod_fft = fft_vecs[1]
    for i in 2:length(msgs)
        prod_fft .*= fft_vecs[i]
    end
    conv_full = irfft(prod_fft, pad_len)

    start_idx = div(pad_len - L, 2) + 1
    msg_out.pdf .= conv_full[start_idx:start_idx + L - 1]
    normalize!(msg_out)

end

convolve_fft!(msg, msgs_scaled)


"""
    interpolate(src, src_len, dst_len)

Linearly interpolate `src` (length src_len) to length dst_len.
"""
function interpolate(src::Vector{Float64}, src_len::Int, dst_len::Int)
    out = zeros(Float64, dst_len)
    scale = (src_len - 1) / (dst_len - 1)
    @inbounds for i in 1:dst_len
        pos = (i - 1) * scale + 1
        i0 = floor(Int, pos)
        i1 = min(i0 + 1, src_len)
        t = pos - i0
        out[i] = (1 - t) * src[i0] + t * src[i1]
    end
    return out
end
