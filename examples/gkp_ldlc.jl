using Distributed

while true
    @everywhere using LatticeDecoder
    @everywhere using LatticeAlgorithms
    @everywhere using LinearAlgebra
    @everywhere using Graphs
    @everywhere using GraphPlot
    @everywhere using BlockArrays
    @everywhere using LLLplus
    @everywhere using LinearAlgebraX
end
# @everywhere using NPZ, LaTeXStrings,LLLplus
# @everywhere using LatticeAlgorithms
# if true
#     @everywhere include(realpath(dirname(@__FILE__))*"/../src/code_constructors/classical_ldlc.jl")

#     # @everywhere include(realpath(dirname(@__FILE__))*"/../src/utilities/utilities.jl")

#     @everywhere include(realpath(dirname(@__FILE__))*"/../src/bp_algorithms/parallel_bp_log_weight.jl")

#     # @everywhere include(realpath(dirname(@__FILE__))*"/../src/bp_algorithms/tanner_graph.jl")
# end

function find_min_nonzero_product(M::Matrix, J::Matrix)
    min_val = Inf
    e_ind = 0
    f_ind = 0
    for j in 1:size(M)[1]
        for k in j:size(M)[1]
            d = M[j, :]' *J* M[k, :]
            if (abs(d) < abs(min_val)) && (d != 0)
                min_val = d
                e_ind = j
                f_ind = k
            end
        end
    end
    @assert min_val != Inf
    return min_val, e_ind, f_ind
end


find_min_nonzero_product([1 2; 0 5],[0 1 ; -1 0])
A = [1 2; 0 5]
A[findall(x->x!=0, A)]

"""
    canonical_basis(M::Matrix{Int})

recursive calculation of the canonical basis of an __integer__ symplectcically lattice

see https://mathoverflow.net/questions/5108/how-do-you-construct-a-symplectic-basis-on-a-lattice
"""
function canonical_basis(M::Matrix{BigInt})
    @assert size(M)[1]%2 == 0 # need even number of vectors
    # base case (deepest recursion)
    if size(M)[2]==4
        return M
    end
    n = Int64.(size(M)[2]/2)
    # define symplectic form
    J = [zeros(BigInt,n,n) BigInt.(Matrix(1I,n,n)); -BigInt.(Matrix(1I,n,n)) zeros(BigInt,n,n)]
    # start accumulating elements for the symplectic basis
    # we look for a potential conjugate pair: needs to have non-zero symplectic product
    # we choose the pair to have product as small as possible
    # when we find a suitable one, we append it to the collected vectors list
    # d is the order of the sublattice
    d, e_ind, f_ind = find_min_nonzero_product(M,J)
    collected = M[[e_ind,f_ind],:]
    if d<0
        M[e_ind,:] = -M[e_ind,:]
        d = -d
    end 


    # compute symplectic products with remaining rows of M
    remaining = M[setdiff(1:size(M)[1], [e_ind,f_ind]),:]
    reduced_dimension = false
    max_trials = 1000
    while !reduced_dimension && max_trials>=0
        products = collected*J*remaining'
        println("products: ", products)
        if all(x->(x==0||x==-0),products) 
            println("case 1")
            # case 1: the remaining vectors are all symplectcically orthogonal to our collected ones
            # recursively canonize the remaining integer_vectors
            remaining = canonical_basis(remaining)
            reduced_dimension = true
        elseif all(x->(x==0||x==-0), products.%d)
            println("case 2")
            # case 2: the remaining vectors are not symplectcically orthogonal to our collected ones
            #   but the symplectic products are all integer multiples of d_1
            for i in 1:size(remaining)[1]
                #   apply "integer" Gram-Smidt and reduce to case 1
                gi = remaining[i,:]
                remaining[i,:] = gi - BigInt(gi'*J*collected[2,:]/d) * collected[1,:] + BigInt(gi'*J*collected[1,:]/d) * collected[2,:]
            end
            println(collected*J*remaining')
            remaining = canonical_basis(remaining)
            reduced_dimension = true
        else
            # case 3: the products <e,g> or <f,g> are !=0 mod d for some of the remaining vectors g
            println("case 3")
            max_trials -= 1
            println("collected ", collected)
            println("remaining ", remaining)
            println("products: ", collected*J*remaining')
            for j in 1:size(collected)[1]
                for l in 1:size(remaining)[1]
                    k = collected[j,:]'*remaining[l,:]
                    if k<0
                        remaining[l,:] = -remaining[l,:] 
                        k = -k
                    end 
                    if k%d != 0
                        q = 1
                        println("k, d = ", k," ",d)
                        # while k-q*d > 0
                        #     q+=1
                        # end
                        replacement = remaining[l,:] - q * collected[j,:]
                        remaining[l,:] = collected[j,:]
                        collected[j,:] = replacement
                        break
                    end
                end
                break
            end

        end
    end
    return vcat(collected,remaining)
end


n = 15
d = 4

J = BigInt.(symplectic_form(n))
J_alt = BigInt.(kron( I(n),symplectic_form(1)))
switch =BigInt.( LatticeAlgorithms.basis_transformation(n))

M = switch*BigInt.(sqrt(d)*classical_ldlc(d, 2*n))
A = M*J_alt*M'

using Primes
factor(det(A))
# factor(det(A))
prod(abs.(eigvals(Int64.(M*J_alt*M'))))
myvec = nullspacex(A'*A-I)
det(A)

M_LLL = (LLLplus.lll(M')[1])'

canonical_M = canonical_basis(M)

canonical_M*J_alt*canonical_M'

adjacency_M = Int64.(Matrix(BlockArray([zeros(size(M)) M ; M' zeros(size(M))])))

display(adjacency_M)

# graph = SimpleDiGraph(adjacency_M)
# gplot(graph)
# simplecycles(graph)

R,canonized_A = LatticeAlgorithms.canonical_form_of_anti_symmetric_matrix(BigInt.(A))
det(R)
display(canonized_A)
canonized_A = switch*canonized_A*switch'
println(abs(detx(M)))

abs.(eigvals(Float64.(A)))
abs.(eigvals(Float64(canonized_A)))

new_M = switch*R*M
new_A = new_M*J*new_M'
display(diag(new_A[1:n,n+1:end]))

new_M_Perp = invx(J*new_M')

display(new_M*J*new_M')

display(new_M_Perp*J*new_M')

display(new_M_Perp*J*new_M'-round.(Int,new_M_Perp*J*new_M'))
display(new_M_Perp*J*new_M_Perp')

println(norm(new_M_Perp[1,1:end]))
println(norm(new_M_Perp[n+1,1:end]))


using LLLplus

B, T = LLLplus.hkz(M)
display(T)
display(M)
switch*R*M*J*M'*R'*switch'
