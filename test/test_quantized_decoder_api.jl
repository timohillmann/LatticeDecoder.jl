using SparseArrays
using Test
using LatticeDecoder

function _small_quantized_ldlc_matrix()
    return sparse([
        1.0 -0.5 0.0;
        0.0 1.0 0.5;
        0.5 0.0 1.0;
    ])
end

function test_quantized_decoder_api()
    H = _small_quantized_ldlc_matrix()
    y = [0.1, -0.2, 0.05]
    sigma = 0.2
    iterations = 2
    quantization = (; L=64, Δ=1 / 16)

    parallel_reference = copy(run_belief_propagation!(
        initialize_tanner_graph_quant(H; quantization...),
        y,
        sigma,
        iterations,
    ))
    parallel_decoder = LDLCDecoder(
        initialize_tanner_graph_quant(H; quantization...);
        schedule=:parallel,
        sigma=sigma,
        max_iterations=iterations,
    )
    parallel_result = run_decoder!(parallel_decoder, y)
    @test parallel_result === parallel_decoder.tg.bp_result
    @test parallel_result ≈ parallel_reference
    @test parallel_decoder.algorithm == :quantized

    serial_reference_graph = initialize_tanner_graph_quant(H; quantization...)
    serial_reference_graph.schedule .= reverse(serial_reference_graph.schedule)
    serial_reference = copy(run_serial_belief_propagation!(
        serial_reference_graph,
        y,
        sigma,
        iterations,
    ))
    for schedule in (:serial, :serial_vertical)
        graph = initialize_tanner_graph_quant(H; quantization...)
        graph.schedule .= reverse(graph.schedule)
        decoder = LDLCDecoder(
            graph;
            schedule=schedule,
            sigma=sigma,
            max_iterations=iterations,
        )
        @test run_decoder!(decoder, y) ≈ serial_reference
    end

    horizontal_reference = copy(LatticeDecoder._run_sommer_horizontal_decoder!(
        initialize_tanner_graph_quant(H; quantization...),
        y,
        sigma,
        iterations,
    ))
    horizontal_decoder = LDLCDecoder(
        initialize_tanner_graph_quant(H; quantization...);
        schedule=:serial_horizontal,
        sigma=sigma,
        max_iterations=iterations,
    )
    @test run_decoder!(horizontal_decoder, y) ≈ horizontal_reference

    @test run_decoder_parallel!(horizontal_decoder, y, sigma, iterations) ≈ parallel_reference
    @test horizontal_decoder.schedule == :parallel

    serial_wrapper_graph = initialize_tanner_graph_quant(H; quantization...)
    serial_wrapper_reference = copy(run_serial_belief_propagation!(
        initialize_tanner_graph_quant(H; quantization...),
        y,
        sigma,
        iterations,
    ))
    serial_wrapper = LDLCDecoder(serial_wrapper_graph; sigma=0.5, max_iterations=0)
    @test run_decoder_serial!(serial_wrapper, y, sigma, iterations) ≈ serial_wrapper_reference
    @test serial_wrapper.schedule == :serial

    string_decoder = LDLCDecoder(
        initialize_tanner_graph_quant(H; quantization...);
        schedule="parallel",
        algorithm="quantized",
        sigma=sigma,
        max_iterations=iterations,
    )
    @test run_decoder!(string_decoder, y) ≈ parallel_reference

    vector_sigma_result = run_belief_propagation!(
        initialize_tanner_graph_quant(H; quantization...),
        y,
        fill(sigma, length(y)),
        iterations,
    )
    @test vector_sigma_result ≈ parallel_reference

    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph_quant(H; quantization...); algorithm=:lsd)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph_quant(H; quantization...); schedule=:layered)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph_quant(H; quantization...); sigma=0.0)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph_quant(H; quantization...); max_iterations=-1)
    @test_throws ArgumentError LDLCDecoder(initialize_tanner_graph_quant(H; quantization...); M=2)
    @test_throws ArgumentError initialize_tanner_graph_quant(H; L=63, Δ=1 / 16)
    @test_throws ArgumentError initialize_tanner_graph_quant(H; L=64, Δ=0.3)

    parallel_decoder.sigma = -1.0
    @test_throws ArgumentError run_decoder!(parallel_decoder, y)
    parallel_decoder.sigma = sigma
    parallel_decoder.max_iterations = -1
    @test_throws ArgumentError run_decoder!(parallel_decoder, y)
end

function test_quantized_decoder_rectangular_and_isolated_nodes()
    H = sparse([1.0 0.0 0.0; 0.0 0.0 0.0])
    graph = initialize_tanner_graph_quant(H; L=64, Δ=1 / 16)
    y = [0.0, 0.125, -0.25]

    @test graph.nc == 2
    @test graph.nv == 3
    @test isempty(graph.check_nodes[2].neighbours)
    @test isempty(graph.var_nodes[2].neighbours)
    @test isempty(graph.var_nodes[3].neighbours)

    result = run_decoder!(LDLCDecoder(graph; sigma=0.2, max_iterations=1), y)
    @test length(result) == length(y)
    @test all(isfinite, result)
    @test result[2:3] ≈ y[2:3]
end

function test_quantized_decoder_high_degree_spectrum_scaling()
    variable_count = 130
    graph = initialize_tanner_graph_quant(
        sparse(ones(1, variable_count));
        L=256,
        Δ=1 / 256,
    )
    result = run_decoder!(
        LDLCDecoder(graph; sigma=0.2, max_iterations=1),
        zeros(variable_count),
    )
    @test length(result) == variable_count
    @test all(isfinite, result)
end

function test_quantized_decoder_allocationless_hot_path()
    repetition_distance = 7
    repetition_code = GKP_Rep_Code(repetition_distance, false, true)
    repetition_H = repetition_code.code[
        (repetition_distance + 1):end,
        (repetition_distance + 1):end,
    ]
    repetition_observations = zeros(repetition_distance)

    for schedule in (:parallel, :serial)
        decoder = LDLCDecoder(
            initialize_tanner_graph_quant(repetition_H; L=256, Δ=1 / 64);
            schedule=schedule,
            sigma=0.25,
            max_iterations=repetition_distance,
        )
        run_decoder!(decoder, repetition_observations)
        run_decoder!(decoder, repetition_observations)
        @test @allocated(run_decoder!(decoder, repetition_observations)) == 0
    end

    general_H = [
        1.0 -0.5 0.25 -0.75;
        0.5 1.0 -0.5 0.25;
    ]
    general_observations = zeros(4)

    for schedule in (:parallel, :serial)
        decoder = LDLCDecoder(
            initialize_tanner_graph_quant(general_H; L=256, Δ=1 / 64);
            schedule=schedule,
            sigma=0.25,
            max_iterations=3,
        )
        run_decoder!(decoder, general_observations)
        run_decoder!(decoder, general_observations)
        @test @allocated(run_decoder!(decoder, general_observations)) == 0
    end
end

function test_quantized_density_kernels()
    graph = initialize_tanner_graph_quant(ones(2, 2); L=64, Δ=1 / 16)
    left = collect(Float64, 1:64)
    right = reverse(left)
    expected = left .* right
    expected ./= sum(expected) * graph.Δ

    output = similar(left)
    LatticeDecoder._multiply_density_to!(output, left, right, graph)
    @test output ≈ expected
    @test sum(output) * graph.Δ ≈ 1.0
    @test @allocated(LatticeDecoder._multiply_density_to!(output, left, right, graph)) == 0

    aliased_output = copy(left)
    LatticeDecoder._multiply_density_to!(aliased_output, aliased_output, right, graph)
    @test aliased_output ≈ expected

    negative_left = copy(left)
    negative_right = copy(right)
    negative_left[1] = -1.0
    negative_right[1] = -1.0
    @test_throws ArgumentError LatticeDecoder._multiply_density_to!(
        output,
        negative_left,
        negative_right,
        graph,
    )

    nonfinite = copy(left)
    nonfinite[1] = NaN
    @test_throws ArgumentError LatticeDecoder._multiply_density_to!(output, nonfinite, right, graph)
    @test_throws ArgumentError LatticeDecoder._multiply_density_to!(output, zeros(64), right, graph)

    stencil_input = [1.0, 2.0, 3.0]
    stencil_output = similar(stencil_input)
    LatticeDecoder._widen_density!(stencil_output, stencil_input)
    @test stencil_output == [3.0, 6.0, 5.0]
    @test_throws ArgumentError LatticeDecoder._widen_density!(stencil_input, stencil_input)

    LatticeDecoder.initialize_messages!(graph, [0.1, -0.05], 0.2)
    LatticeDecoder.check_node_iterations!(graph)
    variable = graph.var_nodes[1]
    expected_messages = Vector{Vector{Float64}}(undef, 2)
    for excluded_edge in 1:2
        incoming = variable.messages[3 - excluded_edge].pdf
        widened = [
            (index == 1 ? 0.0 : incoming[index - 1]) + incoming[index] +
            (index == length(incoming) ? 0.0 : incoming[index + 1])
            for index in eachindex(incoming)
        ]
        expected_message = variable.message.pdf .* widened
        expected_message ./= sum(expected_message) * graph.Δ
        expected_messages[excluded_edge] = expected_message
    end

    LatticeDecoder.variable_node_messages!(graph, 1)
    for edge_index in 1:2
        check_index, _ = variable.neighbours[edge_index]
        check_position = variable.pos_in_check_neighbour[edge_index]
        @test graph.check_nodes[check_index].messages[check_position].pdf ≈
            expected_messages[edge_index]
    end
end
