using LatticeDecoder
using LinearAlgebra
using NPZ
using Random
using SparseArrays
using Test

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

    @test size(a.Mqq) == (size(Hx, 2), size(Hx, 2))
    @test size(a.Mpp) == (size(Hz, 2), size(Hz, 2))
    @test rank(float.(a.Mqq)) == size(Hx, 2)
    @test rank(float.(a.Mpp)) == size(Hz, 2)
    @test a.Mqq == b.Mqq
    @test a.Mpp == b.Mpp

    rank_hx = LatticeDecoder.modp_rank(Hx, 2)
    rank_hz = LatticeDecoder.modp_rank(Hz, 2)
    @test all(x -> iszero(x % 2), a.Mqq[(rank_hx + 1):end, :])
    @test all(x -> iszero(x % 2), a.Mpp[(rank_hz + 1):end, :])
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
