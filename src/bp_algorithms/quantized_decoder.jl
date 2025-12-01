# using FFTW
# using SparseArrays
# using LinearAlgebra
# import Statistics: mean

# FFTW.set_num_threads(Sys.CPU_THREADS)
# # export QuantizedMessage, VariableNodeQuant, CheckNodeQuant, TannerGraphQuant,
# #        initialize_tanner_graph_quant, initialize_from_tanner_graph,
# #        run_belief_propagation_quantized!, check_node_iterations!, variable_node_iterations!,
# #        decision_step!, make_grid

# # -------------------------
# # Message & node types
# # -------------------------
# struct QuantizedMessage
#     pdf::Vector{Float64}   # length L
# end

# mutable struct CheckNodeQuant
#     neighbours::Vector{Tuple{Int64,Float64}}   # (var_index, h_ij)
#     messages::Vector{QuantizedMessage}         # messages from check -> variable (one per neighbour)
#     pos_in_var_neighbour::Vector{Int64}
# end

# mutable struct VariableNodeQuant
#     id::Int64
#     neighbours::Vector{Tuple{Int64,Float64}}   # (check_index, h_ij)
#     message::QuantizedMessage                   # channel message (V's prior)
#     messages::Vector{QuantizedMessage}          # messages from variable -> check (one per neighbour)
#     pos_in_check_neighbour::Vector{Int64}
# end

# # -------------------------
# # Tanner graph for quantized decoder
# # -------------------------
# mutable struct TannerGraphQuant
#     var_nodes::Vector{VariableNodeQuant}
#     check_nodes::Vector{CheckNodeQuant}
#     var_node_to_posit::Dict{Int64,Int64}
#     nv::Int64
#     nc::Int64

#     # quantization params
#     L::Int
#     Δ::Float64
#     grid::Vector{Float64}   # sample points (length L, centered at 0)

#     # runtime
#     bp_result::Vector{Float64}
#     schedule::Vector{Int64}
# end

# # -------------------------
# # Utilities: grid, normalization
# # -------------------------
# """
#     make_grid(Δ::Float64, L::Int)

# Return grid vector of length L centered around 0 with spacing Δ.
# """
# function make_grid(Δ::Float64, L::Int)
#     D = L * Δ
#     half = D / 2
#     return collect(-half:Δ:(half - Δ))
# end

# # safe normalization
# normalize!(v::Vector{Float64}) = (s = sum(v); s > 0 ? (v ./= s) : (v .= 1.0/length(v)))

# # center crop/pad vector `arr` to length `L` (take central L samples if arr longer, pad with zeros otherwise)
# function center_crop(arr::AbstractVector{T}, L::Int) where T
#     n = length(arr)
#     out = zeros(promote_type(T,Float64), L)
#     if n >= L
#         start = max(1, div(n - L, 2) + 1)
#         out .= arr[start:start+L-1]
#     else
#         # place arr centered inside out
#         start = max(1, div(L - n, 2) + 1)
#         out[start:start+n-1] .= arr
#     end
#     return out
# end

# # -------------------------
# # Build TannerGraphQuant
# # -------------------------
# """
#     initialize_tanner_graph_quant(H::SparseMatrixCSC; L=256, Δ=1/64)

# Construct a TannerGraphQuant from parity-check matrix H (size nc × nv).
# `H` must be a sparse matrix in CSC format (rows = checks, cols = variables).
# """
# function initialize_tanner_graph_quant(H::SparseMatrixCSC; L::Int=256, Δ::Float64=1/64)
#     nc, nv = size(H)

#     var_nodes = Vector{VariableNodeQuant}(undef, nv)
#     check_nodes = Vector{CheckNodeQuant}(undef, nc)
#     var_node_to_posit = Dict{Int64,Int64}()

#     node_to_stab = Dict{Int64, Vector{Tuple{Int64,Float64}}}()

#     for c = 1:nc
#         row = H[c, :]
#         nzind = row.nzind
#         nzval = row.nzval
#         neighbours = Vector{Tuple{Int64,Float64}}(undef, length(nzind))
#         pos_in_var_neighbour = Int64[]
#         for k = 1:length(nzind)
#             v = nzind[k]
#             h = nzval[k]
#             neighbours[k] = (v, h)
#             # push into var accumulator
#             if haskey(node_to_stab, v)
#                 push!(node_to_stab[v], (c, h))
#                 push!(pos_in_var_neighbour, length(node_to_stab[v]))
#             else
#                 node_to_stab[v] = [(c, h)]
#                 push!(pos_in_var_neighbour, 1)
#             end
#         end
#         check_nodes[c] = CheckNodeQuant(neighbours, [QuantizedMessage(zeros(Float64,L)) for _ in 1:length(neighbours)], pos_in_var_neighbour)
#     end

#     sorted = sort(collect(node_to_stab), by = x -> x[1])

#     counter = 1
#     for (v, neighs) in sorted
#         var_node_to_posit[v] = counter
#         # channel message placeholder, will be initialized later
#         ch_msg = QuantizedMessage(zeros(Float64, L))
#         # outgoing messages to checks (one per neighbour)
#         msgs = [QuantizedMessage(zeros(Float64,L)) for _ in 1:length(neighs)]
#         var_nodes[counter] = VariableNodeQuant(v, neighs, ch_msg, msgs, Int64[])
#         counter += 1
#     end

#     for c = 1:nc
#         row = H[c,:]
#         for j = 1:length(row.nzind)
#             v = row.nzind[j]
#             vp = var_node_to_posit[v]
#             push!(var_nodes[vp].pos_in_check_neighbour, j)
#         end
#     end

#     grid = make_grid(Δ, L)
#     bp_result = zeros(Float64, nv)
#     schedule = collect(1:nv)

#     return TannerGraphQuant(var_nodes, check_nodes, var_node_to_posit,
#                             nv, nc, L, Δ, grid, bp_result, schedule)
# end

# initialize_tanner_graph_quant(H::Matrix; L::Int=256, Δ::Float64=1/64) = initialize_tanner_graph_quant(sparse(H); L=L, Δ=Δ)



# """
#     expand_pdf_avg(pdf_in, h, Δ)

# Expand the PDF by factor 1/|h| with Sommer–Feder–Shalvi LDLC interpolation.
# This implements:

#    f_exp[k] = (1/(2lw+1)) * sum_{i=-lw..lw} f( (k-i)Δ )

# where lw = floor( ceil(1/|h|) / 2 ).

# """
# function expand_pdf_avg(pdf_in::Vector{Float64}, h::Float64, Δ::Float64)
#     L = length(pdf_in)
#     ah = abs(h)

#     # trivial case (h = ±1): no expansion
#     if ah == 1.0
#         return copy(pdf_in)
#     end

#     # exact window width from LDLC paper:
#     # lw = floor( ceil(1/|h|) / 2 )
#     lw = floor(Int, ceil(1/ah) / 2)
#     lw = max(lw, 1)

#     scale = ah
#     center = (L + 1) / 2
#     res = zeros(Float64, L)

#     @inbounds for k in 1:L
#         # target floating-point input index
#         idxf = ((k - center) / scale) + center
#         idx0 = floor(Int, idxf)   # nearest integer index

#         # averaging window: [il .. ir]
#         il = max(1, idx0 - lw)
#         ir = min(L, idx0 + lw)

#         # compute average without allocations
#         s = 0.0
#         for t in il:ir
#             s += pdf_in[t]
#         end
#         res[k] = s / (ir - il + 1)
#     end

#     # ensure valid PDF
#     normalize!(res)
#     return res
# end


# """
#     convolve_many_fft_sommer(vecs::Vector{Vector{Float64}}, Δ::Float64)

# Compute the LDLC check-node convolution using the Sommer–Feder–Shalvi
# decimated FFT method. Assumes:

#     1/Δ is an integer
#     D = L*Δ is an integer
#     All vectors in `vecs` have identical length L

# Returns a vector of length N = 1/Δ, representing ONE PERIOD of Q̃_j(x),
# which will later be stretched to Q_j(x) = Q̃_j(-h_j x) using interpolation.
# """
# function convolve_many_fft_sommer(vecs::Vector{Vector{Float64}}, Δ::Float64)
#     d = length(vecs)
#     d == 1 && return copy(vecs[1])  # trivial

#     # --- Dimensions ---------------------------------------------------------
#     L = length(vecs[1])
#     @assert all(length(v) == L for v in vecs) "All PDFs must have same length"

#     # number of samples in period-1 representation
#     N = Int(round(1 / Δ))
#     @assert isapprox(N * Δ, 1.0; atol = 1e-12)

#     # block size (number of samples per 1/N segment)
#     D = Int(round(L * Δ))
#     @assert L % D == 0 "L must be divisible by D"
#     @assert D * N == L

#     # --- Precompute block-start offsets for summation -----------------------
#     # For each i = 1..N, samples are f[i], f[i+N], f[i+2N], ..., f[i+(D-1)N]
#     block_offsets = collect(1:N:L)   # length D, constant for all i

#     # --- FFT workspace ------------------------------------------------------
#     # We use rfft because input is real → half-sized FFT
#     fft_len = N


#     # Preallocate FFT arrays
#     fft_arrays = Vector{Vector{ComplexF64}}(undef, d)

#     gi = zeros(Float64, N)         # workspace for block sums

#     # --- Compute decimated FFT for each message -----------------------------
#     @inbounds for j in 1:d
#         f = vecs[j]

#         # Step 1: compute block sums: g[i] = sum_{k=0..D-1} f[i + kN]
#         @inbounds for i in 1:N
#             s = 0.0
#             @simd for b in block_offsets
#                 s += f[i + (b-1)]
#             end
#             gi[i] = s
#         end

#         # Step 2: rFFT of g
#         fft_arrays[j] = rfft(gi)
#     end

#     # --- Multiply FFTs ------------------------------------------------------
#     prod = fft_arrays[1]
#     @inbounds for j in 2:d
#         prod .*= fft_arrays[j]
#     end

#     # --- Return IFFT (one period of Q̃_j(x)) --------------------------------
#     # Output size N (real)
#     conv_period = irfft(prod, fft_len)

#     return conv_period
# end


# """
#     convolve_many_fft_decimated(vecs::Vector{Vector{Float64}}, Δ::Float64)

# Compute the convolution of multiple PDFs `vecs` using FFT with decimation to reduce complexity.

# # Arguments
# - `vecs`: Vector of vectors, each representing an expanded PDF.
# - `Δ`: PDF resolution (should satisfy 1/Δ ∈ ℕ).

# # Returns
# - Vector representing one period of the convolved, stretched PDF.
# """
# function convolve_many_fft_decimated(vecs::Vector{Vector{Float64}}, Δ::Float64)
#     k = length(vecs)
#     k == 1 && return copy(vecs[1])

#     # Number of samples in one period after decimation
#     N = Int(round(1/Δ))
#     d = length(vecs)  # number of PDFs to convolve

#     # Preallocate FFT arrays
#     fft_arrays = Vector{Vector{ComplexF64}}(undef, d)

#     for j in 1:d
#         f = vecs[j]
#         L = length(f)  # length of expanded PDF

#         # Determine decimation factor
#         D = Int(round(L * Δ))  # size of groups to sum
#         @assert L % D == 0 "L must be divisible by D to ensure integer decimation"

#         # Compute group sums (length N = 1/Δ)
#         g = zeros(Float64, N)
#         for i in 1:N
#             @inbounds g[i] = sum(f[i:D:end])  # sum every D-th sample starting at i
#         end

#         # Compute FFT of decimated vector
#         fft_arrays[j] = fft(g)
#     end

#     # Multiply FFTs in frequency domain
#     prod = fft_arrays[1]
#     for j in 2:d
#         prod .*= fft_arrays[j]
#     end

#     # IFFT to get one period of the convolved PDF
#     conv_decimated = real.(ifft(prod))

#     return conv_decimated
# end


# # function convolve_many_fft(vecs::Vector{Vector{Float64}})
# #     k = length(vecs)
# #     k == 1 && return copy(vecs[1])

# #     # Compute convolution length and next power-of-2 FFT size
# #     convlen = sum(length.(vecs)) - (k - 1)
# #     nfft = nextpow(2, convlen)

# #     # Initialize product in frequency domain
# #     prod = ones(ComplexF64, nfft)

# #     # Multiply FFTs of zero-padded vectors
# #     for v in vecs
# #         tmp = ComplexF64.(v)
# #         fftpad!(tmp, nfft)
# #         prod .*= fft(tmp)
# #     end

# #     # Inverse FFT and truncate to convolution length
# #     real.(ifft(prod))[1:convlen]
# # end

# # # Helper: in-place zero-padding to length nfft
# # function fftpad!(v::Vector{ComplexF64}, nfft::Int)
# #     resize!(v, nfft)
# #     v[length(v)+1:end] .= 0
# #     return v
# # end


# # periodic extension: take a single-period Qtilde and make periodic pdf with period = 1/abs(h)
# function periodic_extend_one_period(Qtilde::Vector{Float64}, h::Float64, Δ::Float64, L::Int)
#     period = 1.0 / max(1e-12, abs(h))
#     period_samples = max(1, Int(round(period / Δ)))
#     out = zeros(Float64, L)
#     lenq = length(Qtilde)
#     center = div(L + 1, 2)
#     qcenter = div(lenq + 1, 2)

#     # iterate shifts that could overlap the L window; use range of shifts reasonable to cover support
#     # choose shift multiples around zero that map into [-L..L] samples
#     max_shifts = ceil(Int, L / period_samples) + 2
#     for s = -max_shifts:max_shifts
#         shift = s * period_samples
#         dst_start = center + shift - (qcenter - 1)
#         # add Qtilde at dst_start..dst_start+lenq-1
#         src_i = 1
#         dst_i = dst_start
#         while src_i <= lenq && dst_i <= L
#             if dst_i >= 1
#                 out[dst_i] += Qtilde[src_i]
#             end
#             src_i += 1
#             dst_i += 1
#         end
#     end
#     normalize!(out)
#     return out
# end

# # optional widening (sum neighbors to avoid extreme narrow impulses numerical issues)
# function widen_message(pdf::Vector{Float64})
#     L = length(pdf)
#     tmp = copy(pdf)
#     for k in 1:L
#         left = k>1 ? tmp[k-1] : 0.0
#         right = k<L ? tmp[k+1] : 0.0
#         pdf[k] = left + tmp[k] + right
#     end
#     normalize!(pdf)
#     return pdf
# end

# function _expanded_msg_vector(vn::VariableNodeQuant)

#     msg_vector = Vector{QuantizedMessage}()
#     for i = 1:length(vn.messages)
#         push!(msg_vector, expand(vn.messages[i]))
#     end
#     return msg_vector
# end




# # -------------------------
# # Variable-node update (quantized)
# # -------------------------
# # function variable_node_messages!(tg::TannerGraphQuant, vn_idx::Int)
# #     vn = tg.var_nodes[vn_idx]
# #     d = length(vn.neighbours)
# #     d == 0 && return

# #     for j = 1:d
# #         # outgoing V->C message along edge j
# #         prod = copy(vn.message.pdf)  # start with channel prior

# #         # multiply all incoming C->V messages except edge j
# #         for i = 1:d
# #             if i == j
# #                 continue
# #             end
# #             cn_idx, _ = vn.neighbours[i]
# #             pos = vn.pos_in_check_neighbour[i]
# #             cn = tg.check_nodes[cn_idx]
# #             prod .*= cn.messages[pos].pdf   # read directly from CN
# #         end

# #         normalize!(prod)

# #         # store outgoing message in CN message array
# #         cn_idx, _ = vn.neighbours[j]
# #         pos = vn.pos_in_check_neighbour[j]
# #         tg.check_nodes[cn_idx].messages[pos] = QuantizedMessage(prod)
# #     end
# # end



# # function check_node_messages!(tg::TannerGraphQuant, cn_idx::Int)
# #     cn = tg.check_nodes[cn_idx]
# #     d = length(cn.neighbours)
# #     d == 0 && return

# #     # --- Step 1: Expand all incoming V->C messages --------------------------
# #     expanded = Vector{Vector{Float64}}(undef, d)
# #     hs = Float64[]

# #     for k in 1:d
# #         (v_idx, h) = cn.neighbours[k]
# #         expanded[k] = expand_pdf_avg(cn.messages[k].pdf, h, tg.Δ)
# #         push!(hs, h)
# #     end

# #     # --- Step 2+3: For each outgoing edge j ---------------------------------
# #     for j in 1:d
# #         # gather all expanded except j
# #         toconv = @view expanded[setdiff(1:d, (j,))]

# #         # Sommer convolution → single period of length N = 1/Δ
# #         Q̃ = convolve_many_fft_sommer(toconv, tg.Δ)

# #         # stretch Q_j(x) = Q̃(-h_j * x), result length L
# #         h_j = hs[j]
# #         Qj = stretch_periodic(Q̃, h_j, tg.Δ, tg.L)

# #         # store
# #         cn.messages[j] = QuantizedMessage(Qj)
# #     end

# #     return nothing
# # end


# """
#     stretch_periodic(Q̃, h, Δ, L)

# Given a single-period (length N=1/Δ) representation of  Q̃(x) on [0,1),
# compute samples of Q_j(x) = Q̃(-h x) on the original grid of size L.

# Periodicity is automatic (Q̃ is period-1).
# Linear interpolation is used (can provide cubic if desired).
# """
# function stretch_periodic(Q̃::Vector{Float64}, h::Float64, Δ::Float64, L::Int)
#     N = length(Q̃)
#     @assert N == round(Int, 1/Δ)

#     center = (L+1)/2
#     out = zeros(Float64, L)
#     ah = abs(h)

#     @inbounds for k in 1:L
#         # x_k centered around 0
#         x = (k - center) * Δ                # actual x-coordinate
#         u = -h * x                          # argument into Q̃(x)

#         # because Q̃ has period 1:
#         u = mod(u, 1.0)

#         # convert to Q̃ index space
#         idx = u * N + 1                     # 1-based linear index
#         i0 = floor(Int, idx)
#         t  = idx - i0

#         # wrap
#         i1 = (i0 <= 0)  ? i0 + N : ((i0 > N) ? i0 - N : i0)
#         i2 = i1 == N ? 1 : i1 + 1

#         # linear interpolation
#         out[k] = (1-t)*Q̃[i1] + t*Q̃[i2]
#     end

#     # ensure it's still a PDF
#     normalize!(out)
#     return out
# end


# # iterate all check nodes
# function check_node_iterations!(tg::TannerGraphQuant)
#     for c = 1:tg.nc
#         check_node_messages!(tg, c)
#     end
# end

# # -------------------------
# # Variable-node update (quantized)
# # -------------------------
# function variable_node_messages!(tg::TannerGraphQuant, vn_idx::Int)
#     vn = tg.var_nodes[vn_idx]
#     d = length(vn.neighbours)
#     if d == 0
#         return
#     end

#     for j = 1:d
#         # start with channel prior
#         prod = copy(vn.message.pdf)
#         # multiply incoming check messages except j (they are stored in vn.messages or check_nodes? We assume vn.messages mirrors the V->C storage)
#         for i = 1:d
#             if i != j
#                 prod .*= vn.messages[i].pdf
#             end
#         end
#         normalize!(prod)
#         # assign to corresponding check node's message slot
#         cn_idx, _ = vn.neighbours[j]
#         pos = vn.pos_in_check_neighbour[j]
#         tg.check_nodes[cn_idx].messages[pos] = QuantizedMessage(prod)
#     end
#     return nothing
# end

# # iterate variable nodes
# function variable_node_iterations!(tg::TannerGraphQuant)
#     for v = 1:tg.nv
#         variable_node_messages!(tg, v)
#     end
# end

# # -------------------------
# # Initialization & Decision
# # -------------------------
# """
#     initialize_messages!(tg::TannerGraphQuant, y::Vector{Float64}, σ::Float64)

# Initialize variable (channel) messages from received vector y with noise std σ.
# Also initialize check->var messages as copies of variable priors.
# """
# function initialize_messages!(tg::TannerGraphQuant, y::Vector{Float64}, σ::Float64)
#     @assert length(y) == tg.nv "received vector length must equal number of variables"

#     # build channel PDFs per variable (grid centered at 0; PDF of x with peak at y[v])
#     for vp = 1:tg.nv
#         vnode = tg.var_nodes[vp]
#         # compute gaussian pdf over grid centered at y[vnode.id]
#         pdf = similar(tg.grid)
#         for (i,x) in enumerate(tg.grid)
#             pdf[i] = exp(-0.5 * ((x - y[vnode.id]) / σ)^2)
#         end
#         normalize!(pdf)
#         vnode.message = QuantizedMessage(pdf)
#         # initialize outgoing variable->check messages to channel prior
#         for j = 1:length(vnode.messages)
#             vnode.messages[j] = QuantizedMessage(copy(pdf))
#         end
#     end

#     # initialize check node messages (C->V) as copies of the corresponding variable messages
#     for c = 1:tg.nc
#         cn = tg.check_nodes[c]
#         for j = 1:length(cn.neighbours)
#             vn_idx, _ = cn.neighbours[j]
#             # map variable index to position in var_nodes array
#             vp = tg.var_node_to_posit[vn_idx]
#             # find the pos in variable node's messages that corresponds to this check
#             # in our earlier construction pos_in_check_neighbour holds the position index order per variable
#             pos_in_var = cn.pos_in_var_neighbour[j]
#             cn.messages[j] = QuantizedMessage(copy(tg.var_nodes[vp].messages[pos_in_var].pdf))
#         end
#     end
# end

# # """
# #     decision_step!(tg::TannerGraphQuant)

# # Produce bp_result: the final continuous estimate per variable (grid value at argmax).
# # """
# # function decision_step!(tg::TannerGraphQuant)
# #     for vp = 1:tg.nv
# #         vnode = tg.var_nodes[vp]
# #         finalpdf = copy(vnode.message.pdf)
# #         for m in vnode.messages
# #             finalpdf .*= m.pdf
# #         end
# #         normalize!(finalpdf)
# #         imax = argmax(finalpdf)
# #         tg.bp_result[vnode.id] = tg.grid[imax]
# #     end
# # end

# function decision_step!(tg::TannerGraphQuant)
#     for vp in 1:tg.nv
#         vnode = tg.var_nodes[vp]
#         finalpdf = copy(vnode.message.pdf)

#         for i in 1:length(vnode.neighbours)
#             cn_idx, _ = vnode.neighbours[i]
#             pos = vnode.pos_in_check_neighbour[i]
#             cn = tg.check_nodes[cn_idx]
#             finalpdf .*= cn.messages[pos].pdf
#         end

#         normalize!(finalpdf)
#         imax = argmax(finalpdf)
#         tg.bp_result[vnode.id] = tg.grid[imax]
#     end
# end


# # -------------------------
# # Run loop
# # -------------------------
# """
#     run_belief_propagation_quantized!(tg::TannerGraphQuant, y::Vector{Float64}, σ::Float64; max_iter=tg.max_iters)

# Run quantized LDLC belief-propagation for max_iter iterations (default tg.max_iters).
# Returns tg.bp_result (estimates for each variable).
# """
# function run_belief_propagation_parallel!(tg::TannerGraphQuant, y::Vector{Float64}, σ::Float64, max_iter::Int)
#     initialize_messages!(tg, y, σ)
#     iters = max_iter
#     for it = 1:iters
#         check_node_iterations!(tg)
#         variable_node_iterations!(tg)
#     end

#     decision_step!(tg)
#     return tg.bp_result
# end

# # H = [
# #     0    -0.8   0    -0.5   1     0;
# #     0.8   0     0     1     0    -0.5;
# #     0     0.5   1     0     0.8   0;
# #     0     0    -0.5  -0.8   0     1;
# #     1     0     0     0     0.5   0.8;
# #     0.5  -1    -0.8   0     0     0
# # ]


# # tgq = initialize_tanner_graph_quant(H; L=256, Δ=1/64);
# # using Random
# # y = 0.1 .* randn(6)
# # σ = 0.2
# # res = run_belief_propagation_quantized!(tgq, y, σ; max_iter=20)

# # round.(H * res)


using FFTW
using SparseArrays
using LinearAlgebra

FFTW.set_num_threads(Sys.CPU_THREADS)

# -------------------------
# Message & node types
# -------------------------
struct QuantizedMessage
    pdf::Vector{Float64}   # length L
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

# -------------------------
# Utilities
# -------------------------
make_grid(Δ::Float64, L::Int) = collect(-L*Δ/2 : Δ : (L*Δ/2 - Δ))
normalize!(v::Vector{Float64}) = (s=sum(v); s>0 ? (v ./= s) : (v .= 1/length(v)))

function center_crop(arr::AbstractVector{T}, L::Int) where T
    n = length(arr)
    out = zeros(Float64, L)
    if n >= L
        start = div(n-L,2)+1
        out .= arr[start:start+L-1]
    else
        start = div(L-n,2)+1
        out[start:start+n-1] .= arr
    end
    return out
end

# -------------------------
# Tanner graph constructor
# -------------------------
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
        check_nodes[c] = CheckNodeQuant(neighbours, [QuantizedMessage(zeros(Float64,L)) for _ in neighbours], pos_in_var_neighbour)
    end

    sorted = sort(collect(node_to_stab), by=x->x[1])
    counter = 1
    for (v, neighs) in sorted
        var_node_to_posit[v] = counter
        ch_msg = QuantizedMessage(zeros(Float64,L))
        msgs = [QuantizedMessage(zeros(Float64,L)) for _ in neighs]
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
    return TannerGraphQuant(var_nodes, check_nodes, var_node_to_posit, nv, nc, L, Δ, grid, bp_result)
end

initialize_tanner_graph_quant(H::Matrix; L::Int=256, Δ::Float64=1/64) = initialize_tanner_graph_quant(sparse(H); L=L, Δ=Δ)

# -------------------------
# PDF operations
# -------------------------
function widen_message(pdf::Vector{Float64})
    L = length(pdf)
    tmp = copy(pdf)
    for k in 1:L
        left  = k > 1 ? tmp[k-1] : 0.0
        right = k < L ? tmp[k+1] : 0.0
        pdf[k] = left + tmp[k] + right
    end
    normalize!(pdf)
    return pdf
end


function expand_pdf_avg(pdf_in::Vector{Float64}, h::Float64, Δ::Float64)
    L = length(pdf_in)
    ah = abs(h)
    return ah == 1.0 ? copy(pdf_in) : begin
        lw = max(floor(Int, ceil(1/ah)/2),1)
        scale = ah
        center = (L+1)/2
        res = zeros(Float64,L)
        @inbounds for k in 1:L
            idxf = ((k-center)/scale)+center
            idx0 = floor(Int,idxf)
            il = max(1,idx0-lw)
            ir = min(L,idx0+lw)
            s = sum(pdf_in[il:ir])
            res[k] = s/(ir-il+1)
        end
        normalize!(res)
        res
    end
end

function convolve_many_fft_sommer(vecs::Vector{Vector{Float64}}, Δ::Float64)
    d = length(vecs)
    return d==1 ? copy(vecs[1]) : begin
        L = length(vecs[1])
        N = Int(round(1/Δ))
        D = Int(round(L*Δ))
        @assert L%D==0 && D*N==L
        fft_arrays = Vector{Vector{ComplexF64}}(undef,d)
        gi = zeros(Float64,N)
        block_offsets = collect(1:N:L)
        @inbounds for j in 1:d
            f = vecs[j]
            @inbounds for i in 1:N
                s = 0.0
                for offset in block_offsets
                    s += f[i + (offset - 1)]
                end
                gi[i] = s
            end
            fft_arrays[j] = rfft(gi)
        end
        prod = fft_arrays[1]
        @inbounds for j in 2:d
            prod .*= fft_arrays[j]
        end
        irfft(prod,N)
    end
end

function stretch_periodic(Q̃::Vector{Float64}, h::Float64, Δ::Float64, L::Int)
    N = length(Q̃)
    out = zeros(Float64,L)
    center = (L+1)/2
    @inbounds for k in 1:L
        x = (k-center)*Δ
        u = mod(-h*x,1.0)
        idx = u*N+1
        i0 = floor(Int,idx)
        t = idx-i0
        i1 = i0<=0 ? i0+N : (i0>N ? i0-N : i0)
        i2 = i1==N ? 1 : i1+1
        out[k] = (1-t)*Q̃[i1]+t*Q̃[i2]
    end
    normalize!(out)
    out
end

function periodic_extend_one_period(Q̃::Vector{Float64}, h::Float64, Δ::Float64, L::Int)
    period_samples = max(1, Int(round(1/abs(h)/Δ)))
    out = zeros(Float64,L)
    lenq = length(Q̃)
    center = div(L+1,2)
    qcenter = div(lenq+1,2)
    max_shifts = ceil(Int,L/period_samples)+2
    for s in -max_shifts:max_shifts
        shift = s*period_samples
        dst_start = center + shift - (qcenter-1)
        src_i, dst_i = 1, dst_start
        while src_i <= lenq && dst_i <= L
            if dst_i >= 1
                out[dst_i] += Q̃[src_i]
            end
            src_i += 1
            dst_i += 1
        end
    end
    normalize!(out)
    out
end

# -------------------------
# Check-node updates
# -------------------------
function check_node_messages!(tg::TannerGraphQuant, cn_idx::Int)
    cn = tg.check_nodes[cn_idx]
    d = length(cn.neighbours)
    d==0 && return

    expanded = Vector{Vector{Float64}}(undef,d)
    hs = Float64[]
    for k in 1:d
        (_,h) = cn.neighbours[k]
        expanded[k] = expand_pdf_avg(cn.messages[k].pdf,h,tg.Δ)
        push!(hs,h)
    end

    for j in 1:d
        toconv = @view expanded[setdiff(1:d,(j,))]
        Q̃ = convolve_many_fft_sommer(collect(toconv),tg.Δ)
        h_j = hs[j]
        Qj = stretch_periodic(Q̃,h_j,tg.Δ,tg.L)
        cn.messages[j] = QuantizedMessage(Qj)
    end
end

check_node_iterations!(tg::TannerGraphQuant) = for c in 1:tg.nc; check_node_messages!(tg,c); end


# -------------------------
# Variable-node update (quantized) with widening
# -------------------------
function variable_node_messages!(tg::TannerGraphQuant, vn_idx::Int)
    vn = tg.var_nodes[vn_idx]
    d = length(vn.neighbours)
    if d == 0
        return
    end

    for j = 1:d
        # start with channel prior
        prod = copy(vn.message.pdf)

        # multiply incoming check messages except edge j
        for i = 1:d
            if i != j
                # Widen message to prevent impulse misalignment
                widened = widen_message(vn.messages[i].pdf)
                prod .*= widened
            end
        end

        normalize!(prod)

        # assign to corresponding check node's message slot
        cn_idx, _ = vn.neighbours[j]
        pos = vn.pos_in_check_neighbour[j]
        tg.check_nodes[cn_idx].messages[pos] = QuantizedMessage(prod)
    end
end

variable_node_iterations!(tg::TannerGraphQuant) = for v in 1:tg.nv; variable_node_messages!(tg,v); end

# -------------------------
# Initialization
# -------------------------
function initialize_messages!(tg::TannerGraphQuant, y::Vector{Float64}, σ::Float64)
    @assert length(y)==tg.nv

    for vp in 1:tg.nv
        vnode = tg.var_nodes[vp]
        pdf = [exp(-0.5*((x-y[vnode.id])/σ)^2) for x in tg.grid]
        normalize!(pdf)
        vnode.message = QuantizedMessage(pdf)
        for j in 1:length(vnode.messages)
            vnode.messages[j] = QuantizedMessage(copy(pdf))
        end
    end

    # initialize CN messages from corresponding V->C messages
    for c in 1:tg.nc
        cn = tg.check_nodes[c]
        for j in 1:length(cn.neighbours)
            vn_idx,_ = cn.neighbours[j]
            vp = tg.var_node_to_posit[vn_idx]
            pos = cn.pos_in_var_neighbour[j]
            cn.messages[j] = QuantizedMessage(copy(tg.var_nodes[vp].messages[pos].pdf))
        end
    end
end

# -------------------------
# Decision
# -------------------------
function decision_step!(tg::TannerGraphQuant)
    for vp in 1:tg.nv
        vnode = tg.var_nodes[vp]
        finalpdf = copy(vnode.message.pdf)
        for i in 1:length(vnode.neighbours)
            cn_idx,_ = vnode.neighbours[i]
            pos = vnode.pos_in_check_neighbour[i]
            finalpdf .*= tg.check_nodes[cn_idx].messages[pos].pdf
        end
        normalize!(finalpdf)
        imax = argmax(finalpdf)
        tg.bp_result[vnode.id] = tg.grid[imax]
    end
end

# -------------------------
# Run BP
# -------------------------
function run_belief_propagation_parallel!(tg::TannerGraphQuant, y::Vector{Float64}, σ::Float64, max_iter)
    initialize_messages!(tg,y,σ)
    for it in 1:max_iter
        check_node_iterations!(tg)
        variable_node_iterations!(tg)
    end
    decision_step!(tg)
    return tg.bp_result
end
