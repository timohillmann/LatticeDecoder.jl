using Test
using LatticeDecoder
using LatticeDecoder: TannerGraph, initialize_tanner_graph, initialize_messages!, check_node_iterations!,
    classical_ldlc, _collect_msg_vector, ListSphereDecodingInput, simplified_lsd


# create a small TestCase

struct TestCaseLSD
    σ::Float64
    H::AbstractMatrix
    y::Vector{Float64}
    tg::TannerGraph
end


function TestCaseLSD(H::AbstractMatrix, y::Vector{Float64}, σ::Float64)
    tg = initialize_tanner_graph(H)
    initialize_messages!(tg, y, σ)
    check_node_iterations!(tg)
    return TestCaseLSD(σ, H, y, tg)
end

function test_case_one()
    H = [0.0 -0.8 0.0 -0.5 1.0 0;
        0.8 0.0 0.0 1.0 0.0 -0.5;
        0.0 0.5 1.0 0.0 0.8 0.0;
        0.0 0.0 -0.5 -0.8 0.0 1.0;
        1.0 0.0 0.0 0.0 0.5 0.8;
        0.5 -1.0 -0.8 0.0 0.0 0.0]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    σ = 0.2
    TestCaseOne = TestCaseLSD(H, y, σ)
    return TestCaseOne
end


function test_case_two()
    H = classical_ldlc(5, 961)
    y0 = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = randn(961) / 10
    y[1:5] = y0
    σ = 0.2
    TestCaseTwo = TestCaseLSD(H, y, σ)
    return TestCaseTwo
end


function test_collect_msg_vector_vn_idx(TC::TestCaseLSD, vn_idx::Int)
    vn = TC.tg.var_nodes[vn_idx]
    for nb_idx = 1:length(vn.neighbours)
        msg_vector = _collect_msg_vector(vn, nb_idx)
        @test msg_vector[end] ≈ vn.message
        cnt = 1
        for i = 1:length(msg_vector)
            if i != nb_idx
                @test msg_vector[cnt] ≈ vn.messages[i]
                cnt += 1
            end
        end
    end
end


function test_collect_msg_vector(TC::TestCaseLSD)
    for vn_idx = 1:TC.tg.nv
        test_collect_msg_vector_vn_idx(TC, vn_idx)
    end
end


function test_t_vector(inputs::ListSphereDecodingInput)
    @test sum(abs2, inputs.t_vector) ≈ 1
end

function test_all_t_vectors(TC::TestCaseLSD)
    for vn_idx = 1:TC.tg.nv
        vn = TC.tg.var_nodes[vn_idx]
        for nb_idx = 1:length(vn.neighbours)
            msg_vector = _collect_msg_vector(vn, nb_idx)
            lsd_inputs = ListSphereDecodingInput(msg_vector)
            test_t_vector(lsd_inputs)
        end
    end
end

function test_simplified_lsd(TC::TestCaseLSD)
    for vn_idx = 1:TC.tg.nv
        vn = TC.tg.var_nodes[vn_idx]
        for nb_idx = 1:length(vn.neighbours)
            @testset "lsd_radius_vn_$(vn_idx)_nb_$(nb_idx)" begin
                msg_vector = _collect_msg_vector(vn, nb_idx)
                lsd_inputs = ListSphereDecodingInput(msg_vector)
                test_lsd_radius(lsd_inputs)
            end
        end
    end
end


function test_lsd_radius(lsd_inputs::ListSphereDecodingInput, β_min=1.0, β_max=5.0)
    num_old = 0
    for β = β_min:β_max
        lsd_inputs.β = β
        lsd_inputs.β1 = β
        L, D = simplified_lsd(lsd_inputs)
        @test all(D .<= β^2)
        @test length(D) >= num_old
        num_old = length(D)
    end
end
