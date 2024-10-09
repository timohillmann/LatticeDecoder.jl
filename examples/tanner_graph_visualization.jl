using LatticeDecoder
using Graphs
using GraphPlot
using Compose
using Colors
using LinearAlgebra
using Combinatorics


# Example parity check matrices

H_ring = [
    1 0 -1 ;
    1 1 0;
    0 1 1 
]


H_lin = [
    2 0 0 ;
    1 1 0;
    0 1 1 
]

H_hamming = [
    0  0  0  1  1  1  1;
    0  1  1  0  0  1  1;
    1  0  1  0  1  0  1
    ]

function create_tanner_graph(H)
    m, n = size(H)
    g = SimpleGraph(m + n)
    
    for i in 1:m
        for j in 1:n
            if H[i, j] != 0
                add_edge!(g, i, j + m)
            end
        end
    end
    
    return g
end


function plot_tanner_graph(g, m, n)
    nodefillc = [i <= m ? colorant"lightblue" : colorant"lightgreen" for i in 1:nv(g)]
    
    nodesize = [i <= m ? 0.1 : 0.1 for i in 1:nv(g)]
    
    
    # Create node labels
    labels = vcat(["c$i ($i)" for i in 1:n], ["v$i ($(i+n))" for i in 1:m])

    gplot(g, 
          nodefillc=nodefillc, 
          nodesize=nodesize, # relative node size
          nodelabel=labels,
          layout=spring_layout,
          NODESIZE =0.2) # absolute node size
end

    
function plot_bipartite_tanner_graph(g, m, n)
    # Create node positions
    locs_x = vcat(zeros(m), ones(n))
    locs_y = vcat(range(0, 1, length=m), range(0, 1, length=n))
    
    # Create node labels
    labels = vcat(["c$i ($i)" for i in 1:m], ["v$i ($(i+m))" for i in 1:n])
    
    # Create node colors
    colors = vcat(fill("lightblue", m), fill("lightgreen", n))
    
    # Set different sizes for variable and check nod1s
    # node_sizes = vcat(fill(0.5, n), fill(0.5, m))
    node_sizes = 0.1
    
    
    # Plot the graph
    gplot(g, locs_x, locs_y, nodelabel=labels, nodefillc=colors, nodesize = node_sizes, NODESIZE = 0.15)
end
    


tanner_graph_lin = create_tanner_graph(H_lin)
println(tanner_graph_lin)
m, n = size(H_lin)
plot_tanner_graph(tanner_graph_lin, m, n)
plot_bipartite_tanner_graph(tanner_graph_lin, m, n)

tanner_graph_ring = create_tanner_graph(H_ring)
println(tanner_graph_ring)
m, n = size(H_ring)
plot_tanner_graph(tanner_graph_ring, m, n)
plot_bipartite_tanner_graph(tanner_graph_ring, m, n)

tanner_graph_hamming = create_tanner_graph(H_hamming)
println(tanner_graph_hamming)
m, n = size(H_hamming)
plot_tanner_graph(tanner_graph_hamming, m, n)
plot_bipartite_tanner_graph(tanner_graph_hamming, m, n)

####################################################################
################# counting cycles ##################################
####################################################################

function group_sets_by_length(sets)
    # Initialize an empty dictionary
    result = Dict{Int, Vector{Set}}()
    
    # Iterate through each set in the input
    for set in sets
        length_key = length(set)
        
        # If the length is not already a key, create a new list
        if !haskey(result, length_key)
            result[length_key] = Vector{Set}()
        end
        
        # Add the set to the list corresponding to its length
        push!(result[length_key], set)
    end
    
    return result
end

function find_cycles(g,maximum_length=10, minimum_length = 4, maximum_number = 10^6)
    cycles = Set(Set.(filter(x->length(x)>=minimum_length,simplecycles_limited_length(g, maximum_length, maximum_number))))

    return group_sets_by_length(cycles)
end


# cycles = find_cycles(tanner_graph_lin)
# cycles = find_cycles(ring)
cycles = find_cycles(tanner_graph_hamming)
# Print results
for (length, cycle_list) in cycles
    println("Cycles of length $length:")
    for cycle in cycle_list
        println(cycle)
    end
    println()
end

function compute_girth(g)
    cycles = find_cycles(g)
    if length(cycles) == 0
        println("the graph is a tree or the girth is larger than 10")
        return -1
    else
        return minimum(keys(cycles))
    end
    
end

function compute_girth(H::AbstractMatrix)
    return compute_girth(create_tanner_graph(H))
end

girth =compute_girth(H_lin)
girth =compute_girth(H_ring)
girth =compute_girth(H_hamming)



####################################################################
################# removing dangling checks #########################
####################################################################


function remove_dangling_checks(H::AbstractMatrix)
    # n_checks, n_vars = shape(H)
    dangling_checks = findall(row -> count(!iszero, row) == 1, eachrow(H))
    if length(dangling_checks) == 0 # if no dangling checks we are done
        return H
    end

    higher_weight_checks = findall(row -> count(!iszero, row) > 1, eachrow(H))
    H_new = H[:,:]
    # loop through rows corresponding to dangling checks
    for c in dangling_checks
        println("dangling check $c")
        H_try = H_new[:,:]
        for cp in higher_weight_checks
            println("higher weight check $cp")
            # try adding higher weight rows trying to keep euclidean norm small
            H_try[c,:] = (H_try[c,:]+H[cp,:] < H_try[c,:]-H[cp,:] ? H_try[c,:]+H[cp,:] : H_try[c,:]-H[cp,:])
            girth = compute_girth(H_try)
            println("girth after removal attempt $girth")
            if girth>4 || girth<0
                H_new = H_try[:,:]
            else
                H_try=H_new[:,:]
            end
        end

    end

    dangling_checks = findall(row -> count(!iszero, row) == 1, eachrow(H_new))
    if length(dangling_checks) == 0 # if no dangling checks we are done
        return H_new
    end

    higher_weight_checks = findall(row -> count(!iszero, row) > 1, eachrow(H_new))
    H_new_bis = H_new[:,:]
    combs = combinations(higher_weight_checks,2)
    # loop through rows corresponding to dangling checks
    for c in dangling_checks
        println("dangling check $c")
        H_try = H_new_bis[:,:]
        for comb in combs
            println("higher weight check combination $(comb)")
            for a in Iterators.product(-1:2:1,-1:2:1)
                # try adding higher weight rows
                H_try[c,:] = H_try[c,:] + a[1] * H_new_bis[comb[1],:] + a[2] * H_new_bis[comb[2],:]
        
                girth = compute_girth(H_try)
                println("girth after removal attempt $girth")
                if girth>4 || girth<0
                    H_new_bis = H_try[:,:]
                    println(H_new_bis)
                    break
                else
                    H_try=H_new_bis[:,:]
                end
            end

            println("end a for")
            println(H_new_bis)
            girth = compute_girth(H_new_bis)
            if girth >4 || girth<0 
                println("done with a")
                break
            end
        end

    end



    return H_new_bis
end

H_lin_removed = remove_dangling_checks(H_lin)



println(H_lin_removed)


tanner_graph_lin_removed = create_tanner_graph(H_lin_removed)
println(tanner_graph_lin_removed)
m, n = size(H_lin_removed)
plot_tanner_graph(tanner_graph, m, n)
plot_bipartite_tanner_graph(tanner_graph_lin_removed, m, n)