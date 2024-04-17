include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/tanner_graph.jl")
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/list_sphere_decoder.jl")
include("/Users/timo/Documents/GitHub/LatticeDecoder.jl/src/bp_algorithms/parallel_bp.jl")



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


function test_lsd_radius(lsd_inputs::ListSphereDecodingInput, β_min=1.0, β_max=10.0)
    num_old = 0
    for β = β_min:β_max
        lsd_inputs.β = β
        L, D = simplified_lsd(lsd_inputs)
        @test all(D .<= β^2)
        @test length(D) >= num_old
        num_old = length(D)
    end
end

function _calculate_candidate_gaussians(inputs::ListSphereDecodingInput, L::Vector, D::Vector{Float64}, msg_vector::Vector{gaussian})
    candidate_gaussians = Vector{gaussian}()
    for i = 1:length(L)
        mean = 0.0
        var = inputs.Var
        weight = exp(-1 / 2 * D[i])
        log_weight = -1 / 2 * D[i]

        for j = 1:length(L[i])
            msg = msg_vector[j]
            mean += (msg.mean + L[i][j] / msg.period) / msg.var
        end
        mean *= var
        push!(candidate_gaussians, gaussian(mean, var, weight))
    end
    return candidate_gaussians
end



vn_idx = 1;
nb_idx = 1;
TC = test_case_one();
msg_vector = _collect_msg_vector(TC.tg.var_nodes[vn_idx], nb_idx);
lsd_inputs = ListSphereDecodingInput(msg_vector);

for β in [2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 10.0, 20.0]

    lsd_inputs.β = β
    L, D = simplified_lsd(lsd_inputs)

    candidate_gaussians = _calculate_candidate_gaussians(lsd_inputs, L, D, msg_vector)
    println("β = ", β)
    println("Num gaussians: ", length(L))
    g = moment_matching(candidate_gaussians)
    println("Mean: ", g.mean)
    println("Var: ", g.var)

end

