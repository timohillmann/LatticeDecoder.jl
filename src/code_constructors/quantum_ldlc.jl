# using LinearAlgebra
# using LinearAlgebraX
# using NPZ
# using LLLplus


# """
#     symplectic_form(n::Int)

# Return the symplectic form for `n` modes in the `qqpp` basis.
# """
# function symplectic_form(n::Int)
#     return kron(Float64[0 1; -1 0], Matrix{Float64}(I, n, n))
# end

# function clean_zeros!(M::AbstractArray{Float64}, tol::Float64 = 1e-13)
#     @inbounds @simd for i in eachindex(M)
#         if abs(M[i]) < tol
#             M[i] = 0.0
#         end
#     end
#     return M
# end

# function balance_weights!(mat::AbstractMatrix{Int64}; sign::Int64 = -1)
#     for i in 1:size(mat, 1)
#         row = mat[i, :]
#         num_nzvals = count(!iszero, row)
#         if num_nzvals == 1
#             nz_idx = row.nzind[1]
#             # find the row that has a 1 in the same column
#             for j in 1:size(mat, 1)
#                 if j == i
#                     continue
#                 end
#                 if mat[j, nz_idx] != 0
#                     row2 = mat[j, :]
#                     mat[i, :] .= row2 .+ rand([-1, 1]) * row
#                     break
#                 end
#             end
#         end
#     end
# end

# p = 7
# code_name = "24_8_2_p$(p)"
# code = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/weight_6/$(code_name)_expanded.npz");
# J = symplectic_form(size(code["hx"], 2));
# Z = zeros(size(code["hx"]));
# Mq = code["hx"];
# Mp = code["hz"] ./ sqrt(p);



# using Plots
# heatmap(Mqs, c = cgrad(:rain, rev = false))

# using SparseArrays: sparse, sparse!
# Mqs = sparse(Mq)
# balance_weights!(Mqs)

# using LatticeDecoder
# d = 3
# code = GKP_Surface_Code(d)

# M = code.code
# # M = vcat(hcat(Mq, Z), hcat(Z, Mp));



# # J = code.J
# global num_logicals = 1;
# println("Start")
# for k = 1:size(M, 1)
#     vec = round.(invx(M) *   Mperp[k, :], digits=3) .% 1
#     flag = true
#     for j in eachindex(vec)
#         if vec[j] != 0.0
#             flag = false

#         end
#     end
#     if flag
#         println(k)
#         num_logicals += 1
#     end
# end

# l = M * J * Mperp
# clean_zeros!(l)

# J = code.J
# A = M * J * M';
# Mperp = J * invx(A) * M;
# clean_zeros!(Mperp);

# M

# Minv = - J * M' * J



# flag = false
# for val in res
#     if !(val ≈ 0.0)
#         flag = true
#     end
# end

# flag
# Mperp_T = transpose(Mperp)

# e = zeros(size(M, 1))
# L = code.logical
# for k = 1:100_000
#     e .= rand([0, 1], size(M, 1))
#     res = round.(Int, M * J * e * sqrt(2)) .% 2
#     flagM = any(res .!= 0.0)
#     flagL = any(round.(sqrt(2) * L * e) .% 2.0 .!= 0.0)
#     @assert flagL == flagM "$(e)"
# end
    
# println(res[7]," ", res[10] )

# using BlockDiagonals
# Ω_mat(N) = BlockDiagonal([[[0 1]; [-1 0]] for _ in 1:floor(Int, N/2)])
# Ω = Ω_mat(18)

# clean_zeros!(M * Ω * Mperp)

# clean_zeros!(sqrt(2) * Mperp * e)

# clean_zeros!(sqrt(2) * Mperp_T * e)

# # function css_lattice_distance(Mq::AbstractArray{Float64}, Mp::AbstractArray)
# Apq = round.(BigInt, - Mp * Mq')
# Aqp = round.(BigInt, Mq * Mp')

# Aqp_inv = invx(Aqp)
# Apq_inv = invx(Apq)

# Mq_perp = Aqp_inv * Mq
# Mp_perp = - Apq_inv * Mp


# Mq_perp = 2 * Aqp_inv * Mq
# Mp_perp = - 2 * Apq_inv * Mp




# # function solve_CVP(L::AbstractVector, M::AbstractArray; MAXCVPITER::Int64=100_000_000)
# MAXCVPITER = 100_000_000
# # L = Vector(Mq_perp[1, :])
# # M = Mp



# for k = 1:(2*d^2)
#     L = (Mperp[k, :]);
    
#     m_LLL, Tunimod, _ = LLLplus.lll(M');
#     M_LLL = (M' * Tunimod)';
#     M_LLL_inv = invx(M_LLL);

#     Ls = BigFloat.(L' - round.(L' * M_LLL_inv) * M_LLL)[:];

#     Q, Rtmp = qr(BigFloat.(M_LLL'));
#     R = UpperTriangular(Rtmp);
#     uhat = cvp(Q' * Ls, R, Val(true), -1, 1, MAXCVPITER);

#     val = norm(Ls - M_LLL' * uhat)
#     if val > 0.1
#         println(k, ": ", val)
#     end
# end
# # end




# function reduce(N::Int64, M::AbstractArray, Ucan)
#     # define the canonical generator
#     M = Ucan*M
    
#     # initialize symplectic form
#     J = BigInt.(symplectic_form(N))
    
    
#     M_dual = invx(J*M')
    
#     _, T, _ = LLLplus.lll(BigFloat.(M'))  # transpose convention
#     M_LLL = (M' * T)'
#     M_LLL_inv = invx(qubit_generator_lll)

#     X = Mperp[1, :]
    
    
    
    
    
#     println("\n Constructing LLL-parallelepiped-reduced representatives of qubit logicals...\n")
#     # to find distance of the code, find the shortest representatives of XL, ZL
#     # i.e. solve CVP(XL,qubit_generator)
    
#     # first reduce XL to the fundamental parallelogram see https://quantum-journal.org/papers/q-2022-02-10-648/pdf/ eq 80
#     # inv_qubit_gen = invx(qubit_generator)
    
#     # verify inverse
#     # maximum(abs.(inv_qubit_gen*qubit_generator - Diagonal(ones(N * 2))))
    
#     # find LLL decomposition of the stabilizer generator reduced to a logical qubit (transpose due to column vs row conversion in LLLplus)
#     qubit_generator_lll, unimodT, _ = LLLplus.lll(BigFloat.(qubit_generator'),0.9) # unimodT is the unimodular transformation implementing LLL
#     # unimodT
    
#     qubit_generator_lll = (qubit_generator'*unimodT)' # traspose to get back to row convention (stabilizers are rows)
#     inv_qubit_gen_lll = invx(qubit_generator_lll) # find the inverse
    
#     # verify inverse
#     # maximum(abs.(inv_qubit_gen_lll*qubit_generator_lll - Diagonal(ones(N * 2))))
    
#     # find representatives of logical operators reduced to the fundamental parallelogram of the stabilizer lattice
#     XL_par = (XL' - round.(XL'*inv_qubit_gen_lll)*qubit_generator_lll )[:]
#     ZL_par = (ZL' - round.(ZL'*inv_qubit_gen_lll)*qubit_generator_lll)[:] 
#     YL_par = (ZL'+XL' - round.((ZL'+XL')*inv_qubit_gen_lll)*qubit_generator_lll)[:] 


#     # verify XL_par is a valid representative of XL by checking that they differ by a stabilizer
#     ssX = XL_par-XL
#     u_ssX = ssX'*inv_qubit_gen_lll
    
#     print_if_verbose("the following is zero if the LLL-parallelepiped-reduced XL differs from XL by a stabilizer: ", maximum(abs.(u_ssX-round.(u_ssX))) )
    
    
#     # verify ZL_par is a valid representative of ZL by checking that they differ by a stabilizer
#     ssZ = ZL_par-ZL
#     u_ssZ = ssZ'*inv_qubit_gen_lll
    
#     print_if_verbose("the following is zero if the LLL-parallelepiped-reduced ZL differs from ZL by a stabilizer: ", maximum(abs.(u_ssZ-round.(u_ssZ))) )
    
#     print_if_verbose("\n norm of XL_par: ",@sprintf("%.4f", norm(XL_par)), " and ZL_par: ",@sprintf("%.4f", norm(ZL_par)),"\n")
    
    
#     print_if_verbose("the following are all integer if they commute with qubit stabilizers:")
    
#     print_if_verbose(maximum(abs.((XL-XL_par)'*J*qubit_generator_lll')))
#     print_if_verbose(maximum(abs.((XL-XL_par)'*J*qubit_generator')))
    
#     print_if_verbose(maximum(abs.((ZL-ZL_par)'*J*qubit_generator_lll')))
#     print_if_verbose(maximum(abs.((ZL-ZL_par)'*J*qubit_generator')))
    
    
    
#     ##################################
#     # now we try to find the shortest representatives of logical operators by finding the closest stabilizer displacement to each XL and ZL
#     ##################################
    
#     println("\n finding short XL, ZL representatives solving CVP...\n")
    
    
#     # 1: solve CVP
#     # following lines adapted from https://chrisvwx.github.io/LLLplus.jl/dev/
#     #print(@doc(cvp)) # this prints doc for LLLplus implementation of CVP solver
#     Q, Rtmp=qr(qubit_generator_lll'); 
#     println("QR success")
#     R = UpperTriangular(Rtmp);

#     uhat = nothing # initialize uhat to circumvent namespace issues in try-catch
#     catch_warning_logger = CatchWarningLogger(false)
#     with_logger(catch_warning_logger) do
#         uhat=cvp(Q'*XL_par,R, Val(true),-1,1,
#                  Int64(MAXCVPITER)); ######### WARNING CHANGED NXMAX
#         println("CVP success")
#     end
#     if catch_warning_logger.catched
#         println("cvp failed")
#         throw(error("cvp failed"))
#     end

#     # 2. define the short representative
#     short_XL = XL_par - qubit_generator_lll'*uhat
    
#     # 3. compare norms
#     # norm(XL_par)
#     # norm(short_XL)
    
#     # 4. check commutation with original stabilizers and qubit stabilizers
#     print_if_verbose("following are all zero if the short XL commutes with qubit stabilizers:")
#     print_if_verbose(maximum(abs.(M*J*short_XL - round.(M*J*short_XL))))
#     print_if_verbose(maximum(abs.(qubit_generator*J*short_XL - round.(qubit_generator*J*short_XL))))
#     print_if_verbose(maximum(abs.(qubit_generator_lll*J*short_XL - round.(qubit_generator_lll*J*short_XL))))
    
    
#     # repeat for ZL
    
#     # 1: solve CVP
#     # following lines adapted from https://chrisvwx.github.io/LLLplus.jl/dev/
#     # print(@doc(cvp))
#     # Q, Rtmp=qr(qubit_generator_lll'); 
#     # R = UpperTriangular(Rtmp);

#     uhat = nothing # initialize uhat to circumvent namespace issues in try-catch
#     catch_warning_logger = CatchWarningLogger(false)
#     with_logger(catch_warning_logger) do
#         uhat=cvp(Q'*ZL_par,R, Val(true),-1,1,
#                  Int64(MAXCVPITER)); ######### WARNING CHANGED NXMAX
#     end
#     if catch_warning_logger.catched
#         println("cvp failed")
#         throw(error("cvp failed"))
#     end
#     # 2. define the short representative
#     short_ZL = ZL_par - qubit_generator_lll'*uhat
    
#     # 3. compare norms
#     # norm(ZL_par)
#     # norm(short_ZL)
    
#     # 4. check commutation with original stabilizers and qubit stabilizers
#     print_if_verbose("following are all zero if the short ZL commutes with qubit stabilizers:")
#     print_if_verbose(maximum(abs.(M*J*short_ZL - round.(M*J*short_ZL))))
#     print_if_verbose(maximum(abs.(qubit_generator*J*short_ZL - round.(qubit_generator*J*short_ZL))))
#     print_if_verbose(maximum(abs.(qubit_generator_lll*J*short_ZL - round.(qubit_generator_lll*J*short_ZL))))
    
    

#     # repeat for YL
    
#     # 1: solve CVP
#     # following lines adapted from https://chrisvwx.github.io/LLLplus.jl/dev/
#     # print(@doc(cvp))
#     # Q, Rtmp=qr(qubit_generator_lll'); 
#     # R = UpperTriangular(Rtmp);

#     uhat = nothing # initialize uhat to circumvent namespace issues in try-catch
#     catch_warning_logger = CatchWarningLogger(false)
#     with_logger(catch_warning_logger) do
#         uhat=cvp(Q'*YL_par,R, Val(true),-1,1,
#                  Int64(MAXCVPITER)); ######### WARNING CHANGED NXMAX
#     end
#     if catch_warning_logger.catched
#         println("cvp failed")
#         throw(error("cvp failed"))
#     end
#     # 2. define the short representative
#     short_YL = YL_par - qubit_generator_lll'*uhat
    
#     # 3. compare norms
#     # norm(ZL_par)
#     # norm(short_ZL)
    
#     # 4. check commutation with original stabilizers and qubit stabilizers
#     print_if_verbose("following are all zero if the short ZL commutes with qubit stabilizers:")
#     print_if_verbose(maximum(abs.(M*J*short_YL - round.(M*J*short_YL))))
#     print_if_verbose(maximum(abs.(qubit_generator*J*short_YL - round.(qubit_generator*J*short_YL))))
#     print_if_verbose(maximum(abs.(qubit_generator_lll*J*short_YL - round.(qubit_generator_lll*J*short_YL))))
    
#     print_if_verbose("\n norm of short XL: ",@sprintf("%.4f", norm(short_XL)),", short YL: ",@sprintf("%.4f", norm(short_YL)), " and short ZL: ",@sprintf("%.4f", norm(short_ZL)),"\n")
    
#     # check commutation between short XL, ZL
#     print_if_verbose("The following is half-integer if short XZ and ZL anti-commute:")
#     print_if_verbose(short_XL'*J*short_ZL)
#     # short_XL
#     # short_ZL
    
#     println("\n rounding components of short XL, YL ZL\n")
    
#     # define generator of symplectically dual lattice (logical ops)
#     Mperp = invx(J*qubit_generator_lll')
#     inv_Mperp = invx(Mperp)
#     # snap XL, ZL to the dual lattice
#     short_XL_round = vec(round.(BigInt,short_XL'*inv_Mperp)*Mperp)
#     short_YL_round = vec(round.(BigInt,short_YL'*inv_Mperp)*Mperp)
#     short_ZL_round = vec(round.(BigInt,short_ZL'*inv_Mperp)*Mperp)
    
#     print_if_verbose("the following is half integer if the rounded XL, ZL anti-commute: ",short_XL_round'*J*short_ZL_round)
#     print_if_verbose("the norms of the rounded XL: ", @sprintf("%.4f",norm(short_XL_round)),", rounded YL: ", @sprintf("%.4f",norm(short_YL_round)), " and ZL: ", @sprintf("%.4f",norm(short_ZL_round))," should be the same as before rounding")
    
#     print_if_verbose("the following are zero if the rounding did not dramatically change the short XL, YL, ZL")
#     print_if_verbose(maximum(abs.(short_XL_round-short_XL)))
#     print_if_verbose(maximum(abs.(short_YL_round-short_YL)))
#     print_if_verbose(maximum(abs.(short_ZL_round-short_ZL)))

#     return short_XL_round, short_YL_round, short_ZL_round, qubit_generator, qubit_generator_lll
# end

# #N = 15

# # load the original generator
# #M = npzread("examples/ldlc_gens/gkpldlc_15_2.npy")

# # load the unimodular matrix bringing M in canonical form
# #M = readdlm("examples/ldlc_gens/gkpldlc_15_2_canonical_M.txt", ' ', BigInt)

# #short_XL, short_ZL = reduce(N, M, M)

# #print_if_verbose("short_XL: $(short_XL), short_ZL: $(short_ZL)")