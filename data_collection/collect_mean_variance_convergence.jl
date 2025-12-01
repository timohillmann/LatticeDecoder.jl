using LatticeDecoder
using Statistics
using Plots
using LinearAlgebra
using DelimitedFiles
using NPZ
using LinearAlgebraX
using Nemo

Z2, _ = residue_ring(ZZ, 2)

function extract_mean!(out::AbstractArray{Float64}, tg, iteration)
    for i in 1:length(tg.var_nodes)
        vn = tg.var_nodes[i]
        pos = vn.pos_in_check_neighbour[1]
        cn, weight = vn.neighbours[1]
        out[i, iteration] = tg.check_nodes[cn].messages[pos].mean
    end
end


function extract_variance!(out::AbstractArray{Float64}, tg, iteration::Int64, σ::Float64)
    for i in 1:length(tg.var_nodes)
        vn = tg.var_nodes[i]
        pos = vn.pos_in_check_neighbour[1]
        cn, weight = vn.neighbours[1]
        out[i, iteration] = tg.check_nodes[cn].messages[pos].var / σ^2
    end
end

function random_bitstring!(b::Vector{Int64}, n)
    @inbounds for i = 1:n
        b[i] = rand(0:1)
    end
end

function variable_node_iterations_with_list_size_extraction!(tg::LatticeDecoder.TannerGraph, list_sizes::AbstractArray, iter::Int64)
    for i in 1:length(tg.var_nodes)
        tot_length = 0
        vn = tg.var_nodes[i]
        for j = 1:length(vn.neighbours)
            cn_idx, _ = vn.neighbours[j]
            idx = vn.pos_in_check_neighbour[j]
            cn = tg.check_nodes[cn_idx]
            tot_length += LatticeDecoder._lsd_variable_node_message!(cn.messages[idx], vn, j)
        end
        # list_sizes[i, iter] = tot_length / (length(vn.neighbours))
        # println("Tot length: ", tot_length)
    end
end

function snr_db_to_sigma(snr_db::Float64)
    return 10^(-snr_db / 20)
end


function obtain_masks(tg::LatticeDecoder.TannerGraph)
    MASK = Int[]
    NOT_MASK = Int[]
    for i in 1:length(tg.var_nodes)
        vn = tg.var_nodes[i]
        pos = vn.pos_in_check_neighbour[1]
        weight = vn.messages[1].period
        if abs(weight) >= 0.95
            # printstyled("Weight = $weight", color=:red)
            push!(MASK, i)
        else
            push!(NOT_MASK, i)
        end
    end
    return MASK, NOT_MASK
end

function get_mean_variance_timeseries(H::AbstractArray, Δ::Float64; decoder::String = "nearest", schedule::String = "serial", code_name::String = "None",
    max_iter::Int64 = 100,
    random_encoding::Bool = true,
    search_interval::Float64 = 0.95)
    tg = initialize_tanner_graph(H)
    G = generator_matrix(H)

    tg.search_interval = search_interval

    n = size(H, 1)

    MASK, NOT_MASK = obtain_masks(tg)

    σc = lattice_capacity_std()
    σ = σc * snr_db_to_sigma(Δ)
    message = σ * randn(length(tg.var_nodes))
    var = zeros(length(tg.var_nodes), max_iter)
    means = zeros(length(tg.var_nodes), max_iter)


    #TODO: Consider random message encodings.
    b = zeros(Int64, n)
    if random_encoding
        random_bitstring!(b, n)
        message += encode(b, G)
    end


    # initilization
    LatticeDecoder.initialize_messages!(tg, message, σ)


    if schedule == "serial"
        # basis serial iteration
        if decoder == "lsd"
            vn_update! = LatticeDecoder.update_variable_node_lsd!
        elseif decoder == "nearest"
            vn_update! = LatticeDecoder.update_variable_node_nearest!
        end

        for i = 1:max_iter
            LatticeDecoder.update_reliability_schedule!(tg)
            for vn_idx in tg.schedule
                vn_update!(tg, vn_idx)
                # LatticeDecoder.update_variable_node_lsd!(tg, vn_idx)
                # LatticeDecoder.update_variable_node_nearest!(tg, vn_idx)
            end

            extract_variance!(var, tg, i, σ)
            extract_mean!(means, tg, i)
        end

    
    elseif schedule == "parallel"
        # # basic parallel iteration
        if decoder == "lsd"
            vn_update! = LatticeDecoder.variable_node_iterations_lsd!
        elseif decoder == "nearest"
            vn_update! = LatticeDecoder.variable_node_iterations_nearest!
        end
        
        for i in 1:max_iter
            LatticeDecoder.check_node_iterations!(tg)
            vn_update!(tg)
            # LatticeDecoder.variable_node_iterations_nearest!(tg)
            # LatticeDecoder.variable_node_iterations_lsd!(tg)
            # variable_node_iterations_with_list_size_extraction!(tg, list_sizes, i)
            extract_variance!(var, tg, i, σ)
            extract_mean!(means, tg, i)
        end
    end

    # Subtract encode(b, G) from means so that they are centered around zero
    v = encode(b, G)
    for i in axes(means, 2)
        @views means[:, i] .-= v
    end

    
    # Save results with f-string style filename composition
    filename = "data/timeseries/$(code_name)_$(schedule)_$(decoder).npz"
    npzwrite(filename, Dict("means" => means, "var" => var, "mask" => MASK, "NOT_MASK" => NOT_MASK))
end


function load_classical_ldlc(code_params::Dict)
    d = code_params["d"]
    n = code_params["n"]
    code_name = "ldlc_n$(n)_d$(d)"
    load_dir = "data/classical_ldlc_matrices"
    H = npzread(joinpath(load_dir, "$(code_name)_H.npy"))
    G = npzread(joinpath(load_dir, "$(code_name)_G.npy"))
    return H, G, code_name
end


function load_rep_code(code_params::Dict, balancing::String)
    n = code_params["n"]
    code = GKP_Rep_Code(n, false, true)
    H = code.code[n+1:end, n+1:end]
    if balancing == "standard"
        G = round.(sqrt(2) * inv(H)) / sqrt(2)
        return H, G, "rep_code_$(n)_$(balancing)"
    elseif balancing == "first"
        H[1, 1:2] = [1, -1] / sqrt(2)
        G = round.(sqrt(2) * inv(H)) / sqrt(2)
        return H, G, "rep_code_$(n)_$(balancing)"
    elseif balancing == "last"
        H[1, 1] = 1.0 / sqrt(2)
        H[1, 2] = 0.0 / sqrt(2)
        H[1, end-1] = -1.0 / sqrt(2)
        G = round.(sqrt(2) * inv(H)) / sqrt(2)
        return H, G, "rep_code_$(n)_$(balancing)"
    end
end

function load_surface_code(code_params, balanced::Bool)
    d = code_params["d"]
    n = code_params["n"]
    code = GKP_Surface_Code(d, balanced)
    H = code.code 
    G = round.(sqrt(2) * inv(H)) / sqrt(2)
    return H, G, "surface_code_$(d)_$(balanced)"
end


using SparseArrays: sparse
function load_bb_code(code_name::String, balance_weights::Bool = false)

    p = parse(Int, code_name[end])
    code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_expanded.npz")

    if balance_weights
        H = code_data["hx"]
        H_CSC = sparse(H)
        LatticeDecoder.balance_weights!(H_CSC)
        H = Matrix(H_CSC / sqrt(2))
        G = round.(sqrt(2) * inv(H)) / sqrt(2)
        return H, G, code_name * "_$(balance_weights)"
    end

    H = code_data["hx"] / sqrt(p)
    G = round.(sqrt(2) * inv(H)) / sqrt(2)

    return H, G, code_name * "_$(balance_weights)"
end



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

end

function load_3d_gkp_toric(code_name, balance_weights::Bool)
    Hz = load_toric3d_hz()
    
    if balance_weights
        H = Int.(stack_gkp_generator(Hz))
        H_CSC = sparse(H)
        LatticeDecoder.balance_weights!(H_CSC)
        H = Matrix(H_CSC / sqrt(2))
        G = round.(sqrt(2) * inv(H)) / sqrt(2)
        return H, G, code_name * "_$(balance_weights)"
    end
    
    H = 1 / sqrt(2) * stack_gkp_generator(Hz)
    G = round.(sqrt(2) * inv(H)) / sqrt(2)
    return H, G, code_name * "_$(balance_weights)"
end

codes = [
    # Dict("n" => 128, "d" => 5),
    # Dict("n" => 256, "d" => 5),
    # Dict("n" => 512, "d" => 5),
    # Dict("n" => 1024, "d" => 5),
    # Dict("n" => 9, "d" => 3),
    # Dict("n" => 25, "d" => 5),
    # Dict("n" => 49, "d" => 7),
    # "30_4_5_p2",
    # "48_4_7_p2",
    "toric_3D_3"
    ]

for code in codes
    # Δ = LatticeDecoder.lattice_capacity_std() * 0.7943
    # H, G, code_name = load_classical_ldlc(code)
    # H, G, code_name = load_surface_code(code, false)
    # H, G, code_name = load_bb_code(code, true)
    H, G, code_name = load_3d_gkp_toric(code, false)
    for schedule in ["serial", "parallel"]
        for decoder in ["nearest", "lsd"]
            get_mean_variance_timeseries(H, 1.5; code_name=code_name, decoder=decoder, schedule=schedule)
        end
    end
end
