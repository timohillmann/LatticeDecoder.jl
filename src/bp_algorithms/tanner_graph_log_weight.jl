include("gaussians_log_weight.jl")
using SparseArrays


"""
Define an abstract Node type.
"""
abstract type AbstractNode end


"""
Check node structure contains its state and a set of neighbouring Nodes.
"""
mutable struct CheckNode <: AbstractNode #  do i need this? i could 
    neighbours::Vector{Tuple{Int64,Float64}} # Vector{tuple{Int64, Float64}} # Vector of tuples of the form (check node index, edge weight)
    messages::Vector{gaussian_log_weight}
    pos_in_var_neighbour::Vector{Int64} # Position in the neighbours array of the neighbouring variable node objects
end


"""
Variable node contains it state and a set of neighbouring Nodes.
"""
mutable struct VariableNode <: AbstractNode
    message::gaussian_log_weight  # do I need this?
    id::Int64
    neighbours::Vector{Tuple{Int64,Float64}} # Vector of tuples of the form (check node index, edge weight)
    messages::Vector{gaussian_log_weight}
    pos_in_check_neighbour::Vector{Int64} # Position in the neighbours array of the neighbouring check node objects
end

"""
Tanner graph structure of the code of Int64erest.
"""
mutable struct TannerGraph
    var_nodes::Vector{VariableNode}
    check_nodes::Vector{CheckNode}
    var_node_to_posit::Dict{Int64,Int64}  # the nodes are not ordered by their index in the graph
    nv::Int64
    nc::Int64
    bp_result::Vector{Float64}

    function TannerGraph(var_nodes::Vector{VariableNode}, check_nodes::Vector{CheckNode}, var_node_to_posit::Dict{Int64,Int64})
        new(var_nodes, check_nodes, var_node_to_posit, length(var_nodes), length(check_nodes), Vector{Float64}(undef, length(var_nodes)))
    end

end



"""
    initialize_tanner_graph(H::SparseMatrixCSC{Int6464,Int6464}, max_iter::Int64)

Initialize the Tanner graph for LDLC decoding.

# Arguments
- `H`: The parity-check matrix represented as a sparse matrix in CSC format.
- `max_iter`: The maximum number of iterations for decoding.

# Returns
- None

"""
function initialize_tanner_graph(H::SparseMatrixCSC)
    n, m = size(H) # rows, columns

    tg = TannerGraph(
        Vector{VariableNode}(undef, m),
        Vector{CheckNode}(undef, n),
        Dict{Int64,Int64}(),
    )

    node_to_stab = Dict{Int64,Vector{Tuple{Int64,Float64}}}()


    for i = 1:n
        stab = H[i, :]
        pos_in_var_neighbour = Int64[]
        for vn in 1:length(stab.nzind)
            vn_idx = stab.nzind[vn]
            if vn_idx in keys(node_to_stab)
                push!(node_to_stab[vn_idx], (i, stab.nzval[vn]))
                push!(pos_in_var_neighbour, length(node_to_stab[vn_idx]))
            else
                node_to_stab[vn_idx] = [(i, stab.nzval[vn])]
                push!(pos_in_var_neighbour, 1)
            end
        end
        tg.check_nodes[i] = CheckNode(collect(zip(stab.nzind, stab.nzval)), [gaussian_log_weight(0.0, 1.0) for k = 1:length(stab.nzind)], pos_in_var_neighbour)
    end

    # order node_to_stab by key value
    node_to_stab = sort(collect(node_to_stab), by=x -> x[1])

    counter::Int64 = 1
    for (node, neighbours) in node_to_stab
        tg.var_nodes[counter] = VariableNode(gaussian_log_weight(0.0, 1.0), node, neighbours, [gaussian_log_weight(0.0, 1.0, 0.0, neighbours[k][2]) for k = 1:length(neighbours)], Int64[])
        tg.var_node_to_posit[node] = counter
        counter += 1
    end


    # update location of variable nodes in check node neighbours
    for i in 1:n
        stab = H[i, :]
        for j = 1:length(stab.nzind)
            var_node_index = stab.nzind[j]
            push!(tg.var_nodes[var_node_index].pos_in_check_neighbour, j)
        end
    end
    return tg
end

initialize_tanner_graph(H::Matrix) = initialize_tanner_graph(sparse(H))



# H = Float64[1.01 1.02 0; 0 1.11 1.12; 1.31 0 1.32]; # 3x2 matrix
# H = sparse(H)
# tg = initialize_tanner_graph(H)


