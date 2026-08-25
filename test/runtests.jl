using Test

if "notebooks-only" in ARGS
    include("execute_notebooks.jl")
    @testset "Notebook smoke execution" begin
        test_notebook_smoke()
    end
    exit()
end

include("test_lsd_paper_search.jl")
include("test_gaussians.jl")
include("test_lsd_decoder.jl")
include("test_lattice_osd.jl")
include("test_gkp_generators.jl")
include("test_gaussian_leaves.jl")
include("test_ldlc_decoder_api.jl")
include("test_quantized_decoder_api.jl")
include("test_ldlc_decoder_regression.jl")
include("test_sparse_quantum_codes.jl")
include("test_simulation.jl")
include("publication_audit.jl")
include("execute_notebooks.jl")

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
    test_local_search_streaming_matches_materialized()
    test_local_search_large_orders_are_lazy()
end

@testset "GKP generator completion" begin
    test_stack_gkp_generator()
    test_reduced_gkp_repetition_code()
    test_reduced_gkp_repetition_code_stays_sparse()
end

@testset "Gaussian BP leaves" begin
    test_gaussian_degree_one_checks()
    test_gaussian_degree_one_check_reference_lsd()
    test_gaussian_degree_one_variables()
end



@testset "LDLCDecoder API" begin
    test_ldlc_decoder_api()
    test_ldlc_decoder_api_classical_ldlc_sizes()
end

@testset "Quantized LDLCDecoder API" begin
    test_quantized_decoder_api()
    test_quantized_decoder_rectangular_and_isolated_nodes()
    test_quantized_decoder_high_degree_spectrum_scaling()
    test_quantized_decoder_allocationless_hot_path()
    test_quantized_density_kernels()
end

@testset "LDLCDecoder regression" begin
    test_ldlc_decoder_golden_regression()
end

@testset "Sparse quantum code constructors" begin
    test_construction_a_css_gkp_prime_power()
    test_construction_a_css_gkp_validation()
    test_sparse_quantum_modp_rank()
    test_sparse_quantum_heuristic_enlargement()
    test_sparse_quantum_systematic_enlargement_volume()
    test_sparse_quantum_sparse_echelon_enlargement_volume()
    test_sparse_quantum_trivial_enlargement()
    test_sparse_quantum_dense_npz_roundtrip()
    test_sparse_quantum_balance_weights_on_load()
    test_sparse_quantum_scipy_sparse_fixture()
end

@testset "Reproducible simulation API" begin
    test_binomial_estimate()
    test_classical_simulation_reproducibility()
    test_quantum_simulation_reproducibility()
    test_simulation_validation()
end

@testset "Public release audit" begin
    test_publication_audit()
end

@testset "Notebook smoke execution" begin
    test_notebook_smoke()
end
