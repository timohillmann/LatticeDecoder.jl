# using Distributed

# while true
#     @everywhere using LatticeDecoder
#     # @everywhere using LatticeAlgorithms
#     @everywhere using LinearAlgebra
#     @everywhere using NPZ
#     @everywhere using Primes
# end


using LatticeDecoder
using LinearAlgebra
using NPZ
using Primes
using LinearAlgebraX

d = 4

for n in [100,1000]
    println("n = ", n)
    J = symplectic_form(n)
    j = 1
    while j<=20
        M = Int64.(sqrt(d)*classical_ldlc(d, 2*n))
        # if 2 in Primes.factor(Vector, detx(M))
        if detx(M)%2==0
            npzwrite("examples/ldlc_gens/gkpldlc_$(n)_$(j).npy", M)
            A = M*J*M'
            npzwrite("examples/ldlc_gens/gkpldlc_$(n)_$(j)_A.npy", A)
            println("found $j")
            j+=1
        end
    end
end

run(`sage find_ldlc_qubits.sage`)
