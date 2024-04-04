using Nemo
using LinearAlgebra
using NPZ
using LinearAlgebraX
using DelimitedFiles

Z2, _ = residue_ring(ZZ, 2)

"""
    to_int(x::zzModMatrix)

Converts a `zzModMatrix` to an `Int` matrix.

"""
function to_int(x::zzModMatrix)
    m, n = size(x)
    mat = Int[x[i, j].data for i = 1:m, j = 1:n]
    return mat
end

"""
    dual_hamming_code()

Returns the generator matrix of the dual Hamming code.
    
    """
function dual_hamming_code()
    generator = [1 1 1 0 1 0 0;
        0 1 1 1 0 1 0;
        0 0 1 1 1 0 1]

    return matrix(Z2, generator)
end



"""
    hamming_code()

Returns the generator matrix of the Hamming code.
    
    """
function hamming_code()
    generator = [1 0 1 1 0 0 0;
        0 1 0 1 1 0 0;
        0 0 1 0 1 1 0;
        0 0 0 1 0 1 1]

    return matrix(Z2, generator)
end

"""
    linear_independent_cols(A::AbstractArray)

Returns the indices of the linearly independent columns of the matrix `A`.
    
    """
function linear_independent_cols(A::zzModMatrix)
    m, n = size(A)

    inds = [n]
    c_rank = 1
    for i in (n-1):-1:1
        # check if the rank of the submatrix is increased by adding the i-th column
        _inds = copy(inds)
        push!(_inds, i)

        if rank(A[:, _inds]) > c_rank
            push!(inds, i)
            c_rank += 1
        end
    end

    return inds
end

"""
    linear_independent_cols(A::AbstractArray)

Returns the indices of the linearly independent rows of the matrix `A`.
    
    """
function linear_independent_rows(A::zzModMatrix)
    n,m = size(A)

    inds = [n]
    c_rank = 1
    for i in (n-1):-1:1
        # check if the rank of the submatrix is increased by adding the i-th column
        _inds = copy(inds)
        push!(_inds, i)

        if rank(A[ _inds,:]) > c_rank
            push!(inds, i)
            c_rank += 1
        end
    end

    return inds
end

"""
    stack_gkp_generator(pcm::AbstractArray{Int64})

    Return the generator matrix of the concatenated GKP code.
    
    """
function _stack_gkp_generator(pcm::zzModMatrix)

    m, n = size(pcm)
    inds = linear_independent_cols(pcm)
    # setdiff of 1:n and inds
    gkp_inds = setdiff(1:n, inds)

    r = n - m

    gkp_lattice = zeros(Int128, r, n)
    for i in 1:r
        gkp_lattice[i, gkp_inds[i]] = 2
    end

    pcm_int = to_int(pcm)

    return vcat(gkp_lattice, pcm_int)
end

function _greedy_reduction(pcm::zzModMatrix)
    inds = [1]
    m, n = size(pcm)
    for i in 2:m
        if rank(pcm[inds, :]) < rank(pcm[[inds; i], :])
            push!(inds, i)
        end
    end

    return pcm[inds, :]
end


function get_full_rank_pcm(pcm::zzModMatrix)
    # check that the input has full rank
    rk = rank(pcm)

    if rk == size(pcm, 1)
        return pcm
    else
        pcm = _greedy_reduction(pcm)
    end

    if size(pcm, 1) == rk
        return pcm
    else
        throw(ArgumentError("The input matrix is not full rank"))
    end
end



function stack_gkp_generator(pcm::zzModMatrix)
    # check that the input has full rank
    if rank(pcm) == size(pcm, 1)
        # convert to Int64 using the value property
        return _stack_gkp_generator(pcm)


    else
        pcm = get_full_rank_pcm(pcm)
        return stack_gkp_generator(pcm)
    end

end

function load_hgp_code_Hx()
    hx = npzread("data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_hx.npz")
    hx = Int.(hx)
    return matrix(Z2, hx)
end

function load_hgp_code_Hz()
    hz = npzread("data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_hz.npy")
    hz = Int.(hz)
    return matrix(Z2, hz)
end

function load_lp_code_Hx(N::Integer)
    hx = npzread("data/generator_matrices/binary_codes/small_lp_codes/$(N)_8.npz")["hx"]
    hx = Int.(hx)
    return matrix(Z2, hx)
end

function load_lp_code_Hz(N::Integer)
    hz = npzread("data/generator_matrices/binary_codes/small_lp_codes/$(N)_8.npz")["hz"]
    hz = Int.(hz)
    return matrix(Z2, hz)
end


function load_toric4d_hx()
    # the check matrix is stored as a .txt file. load and convert to matrix
    hx = readdlm("data/generator_matrices/binary_codes/Toric4D_3/hx.txt")
    hx = Int.(hx)
    return matrix(Z2, hx)
end

function load_toric4d_hz()
    # the check matrix is stored as a .txt file. load and convert to matrix
    hz = readdlm("data/generator_matrices/binary_codes/Toric4D_3/hz.txt")
    hz = Int.(hz)
    return matrix(Z2, hz)
end

function load_toric3d_hx()
    # the check matrix is stored as a .txt file. load and convert to matrix
    hx = readdlm("data/generator_matrices/binary_codes/Toric3D_3/hx.txt")
    hx = Int.(hx)
    return matrix(Z2, hx)
end

function load_toric3d_hz()
    # the check matrix is stored as a .txt file. load and convert to matrix
    hz = readdlm("data/generator_matrices/binary_codes/Toric3D_3/hz.txt")
    hz = Int.(hz)
    return matrix(Z2, hz)
end


# Hx = load_toric3d_hx();
# Hz = load_toric3d_hz();

# Hx = dual_hamming_code();
# Hz = dual_hamming_code();

# M_q = 1 / sqrt(2) * stack_gkp_generator(Hx);
# M_p = 1 / sqrt(2) * stack_gkp_generator(Hz);

# num_logicals = 1

# @assert num_logicals ≈ log2(abs(det(M_q))) + log2(abs(det(M_p)))

# println(log2(abs(det(M_q))) + log2(abs(det(M_p))))
# println(log2(abs(det(M_q))))
# println(log2(abs(det(M_p))))

###############################

# Hx =  load_lp_code_Hx(90) ;

# rank(Hx)
# size(Hx)
# full_rank = linear_independent_rows(Hx)
# println(Hx)
# reduced_Hx = Hx[full_rank,:];

# rank(reduced_Hx)
# M_q = 1 / sqrt(2) * stack_gkp_generator(Hx);

# Hz = get_full_rank_pcm(load_lp_code_Hz(90));

# rank(Hz)
# size(Hz)
# # full_rank = linear_independent_rows(Hz)

# M_p = 1 / sqrt(2) * stack_gkp_generator(Hz);


# println(maximum(M_q*M_p'-round.(M_q*M_p')))

# println(log2(abs(det(M_q))) + log2(abs(det(M_p))))



# println(round(abs(det(BigFloat.(M_q))*det(BigFloat.(M_p)))))
# println(det(round.(M_q*M_p')))


# using Primes

# Primes.factor(BigInt.(round(det(M_q*M_p'))))


# unique(M_q*M_p')
# # num_logicals = 80

# using LatticeAlgorithms

# function interleave_matrices(a::Matrix{T}, b::Matrix{T}) where T
#     size(a, 1) == size(b, 1) || error("number of rows mismatch")
#     c = Matrix{T}(undef, 2*size(a, 1), size(a, 2))
#     i = 0
#     for (row_a, row_b) in zip(eachrow(a), eachrow(b))
#         c[i += 1, :] .= row_a
#         c[i += 1, :] .= row_b
#     end
#     return c
# end

# block_Mq = hcat(M_q,zeros(size(M_q)))
# block_Mp = hcat(zeros(size(M_q)), M_p)

# M_other_convention = interleave_matrices(block_Mq,block_Mp)

# J_other_convention = kron(Matrix{Float64}(I,size(M_q)),Matrix([0 1;-1 0]))

# A_other_convention = M_other_convention*J_other_convention*M_other_convention'

# J =  kron(Matrix([0 1;-1 0]),Matrix{Float64}(I,size(M_q)))
# unique(J)

# M_matrix = vcat(block_Mq,block_Mp)

# A = M_matrix*J*M_matrix'

# unique(A)
# println(unique(M_q))

# println("beginning")
# non_in_lattice = 0
# for j in 1:size(M_q)[1]
#     vec =  zeros(Float64, size(M_q)[1])
#     vec[j] = sqrt(2)
#     coords = inv(M_q')*vec
#     int_deviation = abs.(coords-round.(coords))
#     if maximum(int_deviation)> 1e-10
#         non_in_lattice+=1
#         println("j = ", j, " deviation = ",maximum(int_deviation))
#         println("deviating coordinates: ",length(int_deviation[int_deviation .> 1e-10]))
#         println("overlapping coordinates: ",length(coords[coords .> 1e-10]))
#     end
# end
# println("not in lattice: ",non_in_lattice)
# println(size(M_q)[1])

# using NormalForms
# using MatInt

# outMq = BigInt.(sqrt(2)* copy(M_q))
# for j in 1:size(M_q)[1]
#     vec =  zeros(Int64, size(M_q)[1])
#     vec[j] = 2
#     coords = inv(M_q')*vec
#     int_deviation = abs.(coords-round.(coords))
#     if maximum(int_deviation)> 1e-10
#         println(size(vcat(outMq,vec')))
#         # outMq = hnfr!(vcat(outMq,vec'))
#         overlap_indices = findall(x->x>1e-10, coords)
#         smaller_mat = vcat(outMq[overlap_indices,:],vec')
#         smaller_mat =  hermite(vcat(smaller_mat))[1:(end-1),:]
#         outMq[overlap_indices,:] = smaller_mat
#     end
# end
# outMq = Float64.(outMq) ./sqrt(2)

# for j in 1:54
#     vec = outMq[j,:]
#     println(length(vec[vec .> 1e-10]))
# end