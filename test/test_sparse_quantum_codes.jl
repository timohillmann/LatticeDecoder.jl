using LatticeDecoder
using LinearAlgebra
using NPZ
using Random
using SparseArrays
using Test

function test_construction_a_css_gkp_prime_power()
    # 1 + 3 == 0 mod 4, so these are valid ququart CSS checks.
    Hx = sparse([1 1])
    Hz = sparse([1 3])
    construction = construction_a_css_gkp(Hx, Hz; p=2, l=2)

    @test construction.modulus == 4
    @test construction.prime == 2
    @test construction.exponent == 2
    @test construction.scale == 2.0
    @test size(construction.M) == (6, 4)
    @test size(construction.Hx_overcomplete) == (3, 2)
    @test size(construction.Hz_overcomplete) == (3, 2)
    @test issparse(construction.M)
    @test rank(Matrix(construction.M)) == 4
    @test construction.Hx_overcomplete_integer == [1 1; 4 0; 0 4]
    @test construction.Hz_overcomplete_integer == [1 3; 4 0; 0 4]
    @test construction.Hx_balanced_integer == [1 1; -3 1; 1 -3]
    @test construction.Hz_balanced_integer == [1 3; -3 3; 1 -1]
    @test all(count(!iszero, construction.Hx_balanced_integer[row, :]) > 1 for row in 1:3)

    J = symplectic_form(2)
    symplectic_products = Matrix(construction.M * J * transpose(construction.M))
    @test symplectic_products ≈ round.(symplectic_products)
end

function test_construction_a_css_gkp_validation()
    @test_throws ArgumentError construction_a_css_gkp([1 0], [1 0]; p=2, l=1)
    @test_throws ArgumentError construction_a_css_gkp([1 0], [0 1]; p=4, l=1)
    @test_throws ArgumentError construction_a_css_gkp([1 0], [0 1]; p=2, l=0)
    @test_throws DimensionMismatch construction_a_css_gkp([1 0], [1 0 0]; p=2, l=1)
end

function test_sparse_quantum_modp_rank()
    H2 = [1 1 0; 0 1 1; 1 0 1]
    H3 = [1 2 0; 0 1 1; 1 0 2]

    @test LatticeDecoder.modp_rank(H2, 2) == 2
    @test LatticeDecoder.modp_rank(sparse(H2), 2) == 2
    @test LatticeDecoder.modp_rank(H3, 3) == 3
    @test LatticeDecoder.modp_rank(sparse(H3), 3) == 3
end

function test_sparse_quantum_heuristic_enlargement()
    Hx = [1 1 0 0; 0 1 1 0; 1 0 1 0]
    Hz = [1 0 0 1; 0 1 0 1; 0 0 1 1]

    a = enlarge_css_generators(Hx, Hz; p=2, method=:heuristic, max_iters=8, rng=MersenneTwister(17))
    b = enlarge_css_generators(Hx, Hz; p=2, method=:heuristic, max_iters=8, rng=MersenneTwister(17))
    rank_hx = LatticeDecoder.modp_rank(Hx, 2)
    rank_hz = LatticeDecoder.modp_rank(Hz, 2)

    @test size(a.Mqq) == (size(Hx, 2), size(Hx, 2))
    @test size(a.Mpp) == (size(Hz, 2), size(Hz, 2))
    @test rank(float.(a.Mqq)) == size(Hx, 2)
    @test rank(float.(a.Mpp)) == size(Hz, 2)
    @test abs(round(Int, det(float.(a.Mqq)))) == 2^(size(Hx, 2) - rank_hx)
    @test abs(round(Int, det(float.(a.Mpp)))) == 2^(size(Hz, 2) - rank_hz)
    @test a.Mqq == b.Mqq
    @test a.Mpp == b.Mpp

    @test all(x -> iszero(x % 2), a.Mqq[(rank_hx + 1):end, :])
    @test all(x -> iszero(x % 2), a.Mpp[(rank_hz + 1):end, :])
end

function test_sparse_quantum_systematic_enlargement_volume()
    Hx = [1 1 1 0; 0 1 1 1; 1 0 1 1]
    Hz = [1 1 0 1; 1 0 1 1; 0 1 1 1]
    enlarged = enlarge_css_generators(Hx, Hz; p=2, method=:systematic)

    n = size(Hx, 2)
    rank_hx = LatticeDecoder.modp_rank(Hx, 2)
    rank_hz = LatticeDecoder.modp_rank(Hz, 2)

    @test size(enlarged.Mqq) == (n, n)
    @test size(enlarged.Mpp) == (n, n)
    @test abs(round(Int, det(float.(enlarged.Mqq)))) == 2^(n - rank_hx)
    @test abs(round(Int, det(float.(enlarged.Mpp)))) == 2^(n - rank_hz)
    @test log2(abs(det(float.(enlarged.Mqq)))) + log2(abs(det(float.(enlarged.Mpp)))) - n ≈ n - rank_hx - rank_hz
end

function test_sparse_quantum_sparse_echelon_enlargement_volume()
    Hx = sparse([
        1 1 0 1 0
        0 1 1 0 1
        1 0 1 1 0
        0 1 0 1 1
    ])
    Hz = sparse([
        1 0 1 0 1
        1 1 0 1 0
        0 1 1 0 1
        1 0 0 1 1
    ])
    enlarged = enlarge_css_generators(Hx, Hz; p=2, method=:sparse_echelon, progress_io=nothing)

    n = size(Hx, 2)
    rank_hx = LatticeDecoder.modp_rank(Hx, 2)
    rank_hz = LatticeDecoder.modp_rank(Hz, 2)
    logdet_mqq, sign_mqq = logabsdet(float.(enlarged.Mqq))
    logdet_mpp, sign_mpp = logabsdet(float.(enlarged.Mpp))

    @test size(enlarged.Mqq) == (n, n)
    @test size(enlarged.Mpp) == (n, n)
    @test !iszero(sign_mqq)
    @test !iszero(sign_mpp)
    @test logdet_mqq / log(2) ≈ n - rank_hx
    @test logdet_mpp / log(2) ≈ n - rank_hz
    @test (logdet_mqq + logdet_mpp) / log(2) - n ≈ n - rank_hx - rank_hz
end

function test_sparse_quantum_trivial_enlargement()
    Hx = sparse([1 0 1; 0 1 1])
    Hz = sparse([1 1 0; 0 1 1])

    enlarged = enlarge_css_generators(Hx, Hz; p=3, method=:trivial)

    @test enlarged.Mqq == [Matrix(Hx); 3I]
    @test enlarged.Mpp == [Matrix(Hz); 3I]
end

function test_sparse_quantum_dense_npz_roundtrip()
    mktempdir() do dir
        input_path = joinpath(dir, "dense_code.npz")
        output_path = joinpath(dir, "dense_code_enlarged.npz")
        Hx = [1 1 0; 0 1 1]
        Hz = [1 0 1; 1 1 0]
        logicals = [1 0 0]

        npzwrite(input_path, Dict("hx" => Hx, "hz" => Hz, "lx" => logicals))

        loaded = load_sparse_quantum_code(input_path)
        @test loaded.Hx == Hx
        @test loaded.Hz == Hz

        enlarged = write_enlarged_sparse_quantum_code(
            input_path,
            output_path;
            method=:heuristic,
            max_iters=4,
            rng=MersenneTwister(8),
        )
        output = npzread(output_path)

        @test haskey(output, "Mqq")
        @test haskey(output, "Mpp")
        @test output["Mqq"] == enlarged.Mqq
        @test output["Mpp"] == enlarged.Mpp
        @test output["hx"] == Hx
        @test output["hz"] == Hz
        @test output["lx"] == logicals
    end
end

function test_sparse_quantum_balance_weights_on_load()
    mktempdir() do dir
        input_path = joinpath(dir, "dense_code.npz")
        Hx = [1 0 0; 1 1 0; 0 1 1]
        Hz = [0 1 0; 1 1 0; 1 0 1]

        npzwrite(input_path, Dict("hx" => Hx, "hz" => Hz, "label" => [7]))

        loaded_plain = load_sparse_quantum_code(input_path)
        loaded_balanced = load_sparse_quantum_code(input_path; balance_weights=true)

        @test loaded_plain.Hx == Hx
        @test loaded_plain.Hz == Hz
        @test loaded_balanced.Hx == [0 1 0; 1 1 0; 0 1 1]
        @test loaded_balanced.Hz == [1 0 0; 1 1 0; 1 0 1]

        enlarged = enlarge_css_generators(
            loaded_balanced.Hx,
            loaded_balanced.Hz;
            method=:trivial,
        )

        @test enlarged.Mqq[1:size(Hx, 1), :] == loaded_balanced.Hx
        @test enlarged.Mpp[1:size(Hz, 1), :] == loaded_balanced.Hz
        @test loaded_balanced.data["hx"] == Hx
        @test loaded_balanced.data["hz"] == Hz
        @test loaded_balanced.data["label"] == [7]
    end
end

function test_sparse_quantum_scipy_sparse_fixture()
    mktempdir() do dir
        input_path = joinpath(dir, "sparse_code.npz")
        Hx = sparse([1 0 1; 0 1 1])
        Hz = sparse([1 1 0; 1 0 1])

        npzwrite(input_path, Dict(
            "Hx_data" => Int.(nonzeros(Hx)),
            "Hx_indices" => Int.(rowvals(Hx) .- 1),
            "Hx_indptr" => Int.(Hx.colptr .- 1),
            "Hx_shape" => collect(size(Hx)),
            "Hx_format" => Vector{UInt8}(codeunits("csc")),
            "Hz_data" => Int.(nonzeros(Hz)),
            "Hz_indices" => Int.(rowvals(Hz) .- 1),
            "Hz_indptr" => Int.(Hz.colptr .- 1),
            "Hz_shape" => collect(size(Hz)),
            "Hz_format" => Vector{UInt8}(codeunits("csc")),
        ))

        loaded = load_sparse_quantum_code(input_path)

        @test loaded.Hx == Hx
        @test loaded.Hz == Hz
    end
end
