using Test
include("test_lsd_paper_search.jl")
include("test_gaussians.jl")
include("test_lsd_decoder.jl")
include("test_lattice_osd.jl")
include("test_ldlc_decoder_api.jl")
include("test_ldlc_decoder_regression.jl")
include("test_sparse_quantum_codes.jl")

# run test set
@testset "gaussians.jl" begin
    test_gaussian_assignment()
    test_gaussian_divide()
    test_moment_matching()
    test_moment_matching!()
end;

@testset "lsd_decoder.jl" begin
    TC1 = test_case_one()
    test_collect_msg_vector(TC1)
    test_all_t_vectors(TC1)
    test_simplified_lsd(TC1)

    TC2 = test_case_two()
    test_collect_msg_vector(TC2)
    test_all_t_vectors(TC2)
    test_simplified_lsd(TC2)
end;


@testset "correct_number" begin
    for n in 3:10
        for w in 1:3
            for order in [[0, 0, w], [0, w, 0], [w, 0, 0], [w, w - 1, 1]]
                @test test_number_generated(n, order)
            end
        end
    end
end

@testset "LocalSearch full enumeration" begin
    test_local_search_full_enumeration_matches_bruteforce()
    test_local_search_full_enumeration_searches_all_columns()
    test_local_search_full_enumeration_never_worsens()
    test_local_search_constructor_api()
end



@testset "LDLCDecoder API" begin
    test_ldlc_decoder_api()
    test_ldlc_decoder_api_classical_ldlc_sizes()
end

@testset "LDLCDecoder regression" begin
    test_ldlc_decoder_golden_regression()
end

@testset "Sparse quantum code constructors" begin
    test_sparse_quantum_modp_rank()
    test_sparse_quantum_heuristic_enlargement()
    test_sparse_quantum_trivial_enlargement()
    test_sparse_quantum_dense_npz_roundtrip()
    test_sparse_quantum_balance_weights_on_load()
    test_sparse_quantum_scipy_sparse_fixture()
end
