using Test
using Random
using LatticeDecoder

function _small_ldlc_matrix()
    return [
        0.0 -0.8 0.0 -0.5 1.0 0.0;
        0.8 0.0 0.0 1.0 0.0 -0.5;
        0.0 0.5 1.0 0.0 0.8 0.0;
        0.0 0.0 -0.5 -0.8 0.0 1.0;
        1.0 0.0 0.0 0.0 0.5 0.8;
        0.5 -1.0 -0.8 0.0 0.0 0.0;
    ]
end

function test_ldlc_decoder_api()
    H = _small_ldlc_matrix()
    y = [0.1, -0.2, 0.05, 0.3, -0.15, 0.25]
    sigma = 0.2
    iterations = 3

    for schedule in (:parallel, :serial), algorithm in (:nearest, :lsd)
        tg_ref = initialize_tanner_graph(H)
        tg_dec = initialize_tanner_graph(H)

        reference = if schedule == :parallel
            copy(run_belief_propagation!(tg_ref, y, sigma, iterations, String(algorithm)))
        else
            copy(run_serial_belief_propagation!(tg_ref, y, sigma, iterations, String(algorithm)))
        end

        decoder = LDLCDecoder(tg_dec; schedule=schedule, algorithm=algorithm, sigma=sigma, max_iterations=iterations)
        result = run_decoder!(decoder, y)

        @test result === decoder.tg.bp_result
        @test result ≈ reference
    end

    string_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule="parallel", algorithm="nearest", sigma=sigma, max_iterations=iterations)
    string_reference = copy(run_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, "nearest"))
    @test run_decoder!(string_decoder, y) ≈ string_reference
    @test string_decoder.schedule == :parallel
    @test string_decoder.algorithm == :nearest

    m_reference = copy(run_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, 2))
    decoder = LDLCDecoder(initialize_tanner_graph(H), 2)
    decoder.sigma = sigma
    decoder.max_iterations = iterations

    result = run_decoder!(decoder, y)
    @test result === decoder.tg.bp_result
    @test result ≈ m_reference
    @test length(result) == length(y)
    @test decoder.schedule == :parallel
    @test decoder.algorithm == 2

    keyword_m_decoder = LDLCDecoder(initialize_tanner_graph(H); M=2, sigma=sigma, max_iterations=iterations)
    @test run_decoder!(keyword_m_decoder, y) ≈ m_reference
    @test keyword_m_decoder.algorithm == 2

    beta_decoder = LDLCDecoder(initialize_tanner_graph(H); algorithm=:lsd, sigma=sigma, max_iterations=iterations, lsd_beta=2.0)
    beta_reference = copy(run_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, "lsd"; lsd_beta=2.0))
    @test run_decoder!(beta_decoder, y) ≈ beta_reference
    @test beta_decoder.lsd_beta == 2.0
    @test beta_decoder.tg.lsd_beta == 2.0

    default_w_min_decoder = LDLCDecoder(initialize_tanner_graph(H); algorithm=:lsd, sigma=sigma, max_iterations=iterations)
    explicit_w_min_decoder = LDLCDecoder(initialize_tanner_graph(H); algorithm=:lsd, sigma=sigma, max_iterations=iterations, lsd_w_min=LatticeDecoder.LSD_W_MIN)
    @test run_decoder!(explicit_w_min_decoder, y) ≈ run_decoder!(default_w_min_decoder, y)

    w_min_decoder = LDLCDecoder(initialize_tanner_graph(H); algorithm=:lsd, sigma=sigma, max_iterations=iterations, lsd_beta=2.0, lsd_w_min=0.5)
    w_min_reference = copy(run_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, "lsd"; lsd_beta=2.0, lsd_w_min=0.5))
    @test run_decoder!(w_min_decoder, y) ≈ w_min_reference
    @test w_min_decoder.lsd_w_min == 0.5
    @test w_min_decoder.tg.lsd_w_min == 0.5

    memory_zero_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=:serial, algorithm=:lsd, sigma=sigma, max_iterations=iterations, memory_strength=0.0)
    @test run_decoder!(memory_zero_decoder, y) ≈ run_serial_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, "lsd")

    damping_zero_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=:parallel, algorithm=:nearest, sigma=sigma, max_iterations=iterations, damping_strength=0.0)
    @test run_decoder!(damping_zero_decoder, y) ≈ run_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, "nearest")

    for schedule in (:parallel, :serial)
        memory_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=schedule, algorithm=:nearest, sigma=sigma, max_iterations=iterations, memory_strength=0.25)
        memory_result = run_decoder!(memory_decoder, y)
        @test memory_result === memory_decoder.tg.bp_result
        @test all(isfinite, memory_result)
        @test [msg.mean for msg in memory_decoder.channel_messages] ≈ y

        vector_memory_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=schedule, algorithm=:lsd, sigma=sigma, max_iterations=iterations, memory_strength=fill(0.1, length(y)))
        vector_memory_result = run_decoder!(vector_memory_decoder, y)
        @test vector_memory_result === vector_memory_decoder.tg.bp_result
        @test all(isfinite, vector_memory_result)

        negative_memory_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=schedule, algorithm=:nearest, sigma=sigma, max_iterations=iterations, memory_strength=-0.25)
        negative_memory_result = run_decoder!(negative_memory_decoder, y)
        @test negative_memory_result === negative_memory_decoder.tg.bp_result
        @test all(isfinite, negative_memory_result)

        negative_vector_memory_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=schedule, algorithm=:lsd, sigma=sigma, max_iterations=iterations, memory_strength=fill(-0.1, length(y)))
        negative_vector_memory_result = run_decoder!(negative_vector_memory_decoder, y)
        @test negative_vector_memory_result === negative_vector_memory_decoder.tg.bp_result
        @test all(isfinite, negative_vector_memory_result)

        damping_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=schedule, algorithm=:nearest, sigma=sigma, max_iterations=iterations, damping_strength=0.35)
        damping_result = run_decoder!(damping_decoder, y)
        @test damping_result === damping_decoder.tg.bp_result
        @test all(isfinite, damping_result)

        vector_damping_decoder = LDLCDecoder(initialize_tanner_graph(H); schedule=schedule, algorithm=:lsd, sigma=sigma, max_iterations=iterations, damping_strength=fill(0.15, length(y)))
        vector_damping_result = run_decoder!(vector_damping_decoder, y)
        @test vector_damping_result === vector_damping_decoder.tg.bp_result
        @test all(isfinite, vector_damping_result)
    end

    decoder.sigma = 0.25
    decoder.max_iterations = 1
    @test run_decoder!(decoder, y) === decoder.tg.bp_result

    serial_result = run_decoder_serial!(decoder, y, sigma, iterations)
    @test serial_result === decoder.tg.bp_result
    @test decoder.schedule == :serial

    parallel_result = run_decoder_parallel!(decoder, y, sigma, iterations)
    @test parallel_result === decoder.tg.bp_result
    @test decoder.schedule == :parallel

    memory_wrapper_result = run_decoder_serial!(decoder, y, sigma, iterations; memory_strength=-0.2, damping_strength=0.3)
    @test memory_wrapper_result === decoder.tg.bp_result
    @test decoder.memory_strength == -0.2
    @test decoder.damping_strength == 0.3

    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); schedule=:layered)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); schedule=1)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); algorithm=:paper_lsd)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); algorithm=2.5)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); algorithm=0)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); algorithm=:lsd, lsd_beta=0.0)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); algorithm=:lsd, lsd_w_min=-0.1)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); memory_strength=-1.1)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); memory_strength=1.1)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); memory_strength=fill(0.1, length(y) - 1))
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); damping_strength=-0.1)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); damping_strength=1.1)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph(H); damping_strength=fill(0.1, length(y) - 1))
    @test_throws ArgumentError run_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, "lsd"; lsd_beta=0.0)
    @test_throws ArgumentError run_belief_propagation!(initialize_tanner_graph(H), y, sigma, iterations, "lsd"; lsd_w_min=-0.1)
end

function test_ldlc_decoder_api_classical_ldlc_sizes()
    sigma = 0.2

    for (n, degree, seed) in ((256, 5, 2565), (986, 7, 9867))
        Random.seed!(seed)
        H = classical_ldlc(degree, n)
        y = [0.1 * sin(i) for i in 1:n]

        iterations = 2

        for algorithm in (:nearest, :lsd)
            tg_ref = initialize_tanner_graph(H)
            tg_dec = initialize_tanner_graph(H)

            reference = copy(run_belief_propagation!(tg_ref, y, sigma, iterations, String(algorithm)))
            decoder = LDLCDecoder(tg_dec; schedule=:parallel, algorithm=algorithm, sigma=sigma, max_iterations=iterations)
            result = run_decoder!(decoder, y)

            @test result === decoder.tg.bp_result
            @test length(result) == n
            @test all(isfinite, result)
            @test result ≈ reference
        end

        m_iterations = 1
        tg_m_ref = initialize_tanner_graph(H)
        m_reference = copy(run_belief_propagation!(tg_m_ref, y, sigma, m_iterations, 2))

        m_decoder = LDLCDecoder(initialize_tanner_graph(H), 2)
        m_decoder.sigma = sigma
        m_decoder.max_iterations = m_iterations
        m_result = run_decoder!(m_decoder, y)

        @test m_result === m_decoder.tg.bp_result
        @test length(m_result) == n
        @test all(isfinite, m_result)
        @test m_result ≈ m_reference
        @test m_decoder.schedule == :parallel
        @test m_decoder.algorithm == 2
    end
end
