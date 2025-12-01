using LatticeDecoder
using Statistics
using Plots
using LinearAlgebra
using DelimitedFiles
using NPZ


# d = 6;
# n = 354;


# H = classical_ldlc(d, n, true);
# G = generator_matrix(H);
# tg = initialize_tanner_graph(H);

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


"""
    variable_node_iterations!(tg::TannerGraph)

    Iterates over all variable nodes and updates the messages of the check nodes.
"""
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


# MASK = Int[]
# NOT_MASK = Int[]
# for i in 1:length(tg.var_nodes)
#     vn = tg.var_nodes[i]
#     pos = vn.pos_in_check_neighbour[1]
#     weight = vn.messages[1].period
#     if abs(weight) >= 1.0
#         # printstyled("Weight = $weight", color=:red)
#         push!(MASK, i)
#     else
#         push!(NOT_MASK, i)
#     end
# end

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
        if abs(weight) >= 1.0
            # printstyled("Weight = $weight", color=:red)
            push!(MASK, i)
        else
            push!(NOT_MASK, i)
        end
    end
    return MASK, NOT_MASK
end  

# using Plots
# max_iter = 100

# fig = plot()
# plot!(fig, ones(max_iter), lw=1, linestyle=:dash, color=:black)
# for Δ ∈ [1.5]

#     σc = lattice_capacity_std()
#     σ = σc * snr_db_to_sigma(Δ)
#     println(σ)
#     message = σ * randn(length(tg.var_nodes))
#     var = zeros(length(tg.var_nodes), max_iter)
#     means = zeros(length(tg.var_nodes), max_iter)
#     list_sizes = zeros(Float64, length(tg.var_nodes), max_iter)

#     b = zeros(Int64, n)
#     # b[1] = 1
#     # b[4] = 1
#     # message += encode(b, G)

#     # y = encode(b, G);

#     # g  = decode(y, H);

#     # initilization
#     LatticeDecoder.initialize_messages!(tg, message, σ)

#     # # basic parallel iteration
#     for i in 1:max_iter
#         LatticeDecoder.check_node_iterations!(tg)
#         LatticeDecoder.variable_node_iterations_nearest!(tg)
#         # LatticeDecoder.variable_node_iterations_lsd!(tg)
#         # variable_node_iterations_with_list_size_extraction!(tg, list_sizes, i)
#         extract_variance!(var, tg, i, σ)
#         extract_mean!(means, tg, i)
#     end


#     # # basis serial iteration
#     # for i = 1:max_iter
#     #     LatticeDecoder.update_reliability_schedule!(tg)
#     #     for vn_idx in tg.schedule
#     #         LatticeDecoder.update_variable_node_lsd!(tg, vn_idx)
#     #         # LatticeDecoder.update_variable_node_nearest!(tg, vn_idx)
#     #     end

#     #     extract_variance!(var, tg, i, σ)
#     #     extract_mean!(means, tg, i)
#     # end

#     # decision_step(tg)

#     # H[4, :]' * tg.bp_result

#     # # plot variance of message vs iteration
#     fig = plot(var[MASK, :]', label="", xlabel="iteration", ylabel="variance of message", title="Variance of message vs iteration",
#         lw=0.1, color=2)
#     plot!(fig, mean(var[MASK, :]', dims=2), label="Avg(Mean)", color=:red)

#     plot!(fig, var[NOT_MASK, :]', label="", xlabel="iteration", ylabel="variance of message", title="Variance of message vs iteration",
#         lw=0.1, color=1)
#     plot!(fig, mean(var[NOT_MASK, :]', dims=2), label="Avg(Var)", color=:blue)

#     # # cut of plot y axis at 0.5
#     fig = plot!(fig, ylims=(1e-7, 0.75), yscale=:log10)

#     # # plot 1/(1.6 * iteration)
#     # plot!(1 ./ (1.6 * (1:max_iter)), label="1/(1.6 * iteration)", lw=2,
#     #     marker=:circle, markersize=5)

#     # #plot horizontal line at 2/3
#     # plot!(2 / 3 * ones(max_iter), label="2/3", lw=2, linestyle=:dash, color=:black)

#     # # make new figure for the means
#     # fig = plot()
#     # # plot means of message vs iteration
#     # fig = plot(means', label="", xlabel="iteration", ylabel="mean of message", title="Mean of message vs iteration")


#     #plot!(fig, list_sizes', label="", xlabel="iteration", ylabel="list size", title="List size vs iteration")
#     # plot the mean of list size
#     # plot!(fig, (sum(list_sizes, dims=1) / size(list_sizes, 1))', label="Δ = $(Δ)", lw=1, marker=:circle, markersize=2)
#     # plot horizontal dashed line at 1
#     # using BenchmarkTools

#     npzwrite("data_collection/random_ldlc_serial_lsd.npz", Dict("means" => means, "var" => var, "mask" => MASK, "NOT_MASK" => NOT_MASK))
# end

# display(fig)


function get_mean_variance_timeseries(H::AbstractArray, Δ::Float64; decoder::String = "nearest", schedule::String = "serial", code_name::String = "None",
    random_encoding::Bool = false)
    tg = initialize_tanner_graph(H)
    G = generator_matrix(H)

    n = size(H, 1)

    MASK, NOT_MASK = obtain_masks(tg)

    σc = lattice_capacity_std()
    σ = σc * snr_db_to_sigma(Δ)
    println(σ)
    message = σ * randn(length(tg.var_nodes))
    var = zeros(length(tg.var_nodes), max_iter)
    means = zeros(length(tg.var_nodes), max_iter)


    #TODO: Consider random message encodings.
    if random_encoding
        b = zeros(Int64, n)
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
        elseif decoder "nearest"
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


# get_mean_variance_timeseries(H, 1.5; code_name="random_ldlc")