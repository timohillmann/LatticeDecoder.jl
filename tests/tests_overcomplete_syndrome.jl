using Test
using LatticeDecoder

function gkp_rep_code(d::Int, bit_flip=false)
    H = zeros(Float64, d - 1, d)
    for i in 1:(d-1)
        H[i, i] = 1
        H[i, i+1] = 1

    end

    if bit_flip
        M_H = [H zeros(Float64, d - 1, d)]
    else
        M_H = [zeros(Float64, d - 1, d) H]
    end


    GKP_generators = Matrix{Float64}(2 * I, 2 * d, 2 * d)

    return [GKP_generators; M_H] / sqrt(2)

end

function symplectic_form(n::Int)
    return kron([0 1; -1 0], Matrix{Float64}(I, n, n))
end

function generate_error(n::Int, σ::Float64)
    return σ * randn(2 * n)
end


function compute_syndrome(M::AbstractMatrix{Float64}, J::AbstractMatrix{Float64}, e::AbstractVector{Float64})
    return M * J * e .% 1
end


function compute_eta(M_r::AbstractMatrix{Float64}, J::AbstractMatrix{Float64}, Vt::AbstractMatrix, s::AbstractVector, z::AbstractVector)
    return inv(M_r*J) * Vt * (s + z)
end 

d = 3
M = gkp_rep_code(d)
J = symplectic_form(d)
e = generate_error(d, 0.1)
s = compute_syndrome(M, J, e)

Mr, Vt, V = overcomplete_syndrome_preperation(M)
z = integer_solve(V, s)

η = compute_eta(Mr, J, Vt, s, z)

M * J * (e - η) 


function test_overcomplete_syndrome()

    n_errors = 100
    d = 5 
    M = gkp_rep_code(d)
    J = symplectic_form(d)
    Mr, Vt, V = overcomplete_syndrome_preperation(M)