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
    n, m = size(A)

    inds = [n]
    c_rank = 1
    for i in (n-1):-1:1
        # check if the rank of the submatrix is increased by adding the i-th column
        _inds = copy(inds)
        push!(_inds, i)

        if rank(A[_inds, :]) > c_rank
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

