using Test
using LatticeDecoder

function test_gaussian_degree_one_checks()
    sigma = 0.1

    for schedule in (:parallel, :serial_vertical, :serial_horizontal),
        edge_weight in (0.5, 1.0, 2.0, -2.0),
        observation in (-1.2, 1.2)

        H = reshape([edge_weight], 1, 1)
        nearest = run_decoder!(
            LDLCDecoder(
                initialize_tanner_graph(H);
                schedule=schedule,
                algorithm=:nearest,
                sigma=sigma,
                max_iterations=1,
            ),
            [observation],
        )[1]
        lsd = run_decoder!(
            LDLCDecoder(
                initialize_tanner_graph(H);
                schedule=schedule,
                algorithm=:lsd,
                sigma=sigma,
                max_iterations=1,
            ),
            [observation],
        )[1]
        m_gaussian = run_decoder!(
            LDLCDecoder(
                initialize_tanner_graph(H);
                schedule=schedule,
                algorithm=2,
                sigma=sigma,
                max_iterations=1,
            ),
            [observation],
        )[1]

        @test lsd ≈ nearest atol=2e-6
        @test m_gaussian ≈ nearest atol=2e-6
        @test round(Int, edge_weight * lsd) == round(Int, edge_weight * observation)
    end
end

function test_gaussian_degree_one_check_reference_lsd()
    H = reshape([2.0], 1, 1)
    observation = [1.2]
    graph = initialize_tanner_graph(H)
    LatticeDecoder.initialize_messages!(graph, observation, 0.1)
    LatticeDecoder.check_node_iterations!(graph)

    result = zeros(1)
    LatticeDecoder._lsd_variable_node_decision_reference!(result, graph, 1)
    @test result[1] ≈ 1.037928986626065 atol=2e-6
    @test round(Int, H[1] * result[1]) == round(Int, H[1] * observation[1])
end

function test_gaussian_degree_one_variables()
    H = reshape([1.0, -0.5], 1, 2)
    observations = [0.3, -0.2]
    sigma = 0.15

    for schedule in (:parallel, :serial_vertical, :serial_horizontal), algorithm in (:nearest, :lsd, 2)
        decoder = LDLCDecoder(
            initialize_tanner_graph(H);
            schedule=schedule,
            algorithm=algorithm,
            sigma=sigma,
            max_iterations=1,
        )
        run_decoder!(decoder, observations)

        for edge_index in eachindex(decoder.tg.check_nodes[1].messages)
            outgoing = decoder.tg.check_nodes[1].messages[edge_index]
            @test outgoing.mean ≈ observations[edge_index]
            @test outgoing.var ≈ sigma^2
        end
    end
end
