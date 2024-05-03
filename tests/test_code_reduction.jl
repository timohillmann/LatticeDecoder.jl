using Test
using LatticeDecoder
using DelimitedFiles
using NPZ
using LinearAlgebraX
using Nemo

Z2, _ = residue_ring(ZZ, 2)

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


function toric_4d_test_set()
    Hx = load_toric4d_hx()
    Hz = load_toric4d_hz()
    M_q = 1 / sqrt(2) * stack_gkp_generator(Hx)
    M_p = 1 / sqrt(2) * stack_gkp_generator(Hz)
    num_logicals = 6

    @test num_logicals ≈ log2(abs(det(M_q))) + log2(abs(det(M_p)))
end

function toric_3d_test_set()
    Hx = load_toric3d_hx()
    Hz = load_toric3d_hz()

    M_q = 1 / sqrt(2) * stack_gkp_generator(Hx)
    M_p = 1 / sqrt(2) * stack_gkp_generator(Hz)

    num_logicals = 3

    @test num_logicals ≈ log2(abs(det(M_q))) + log2(abs(det(M_p)))
end
