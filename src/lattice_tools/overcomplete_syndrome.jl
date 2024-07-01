using LinearAlgebra
using NormalForms
using LLLplus

struct BottomSystem
    Vb::Matrix{Int64}
    VbH::ColumnHermite
end

"""
    overcomplete_syndrome_preperation(M::Matrix{Int64})

Prepares the syndrome matrix for the overcomplete syndrome decoding algorithm. The syndrome matrix is transformed into a Hermite normal form and the bottom part of the transformation matrix is returned. The bottom part of the transformation matrix is used to solve the syndrome equation.

"""
function overcomplete_syndrome_preperation(M::Matrix{Int64})
    Mh, U = hnfr(M)
    
    # reduce Mh to non-zero rows
    Mh = Mh[1:size(Mh, 2), :]

    Vt = U[1:size(Mh, 2), :]
    Vb = U[size(Mh, 2)+1:end, :]
    VbH = hnfc(Vb)
    return Mh / sqrt(2), Vt, BottomSystem(Vb, VbH)
end


"""
    overcomplete_syndrome_preperation(M::Matrix{Float64})

Prepares the syndrome matrix for the overcomplete syndrome decoding algorithm. The syndrome matrix is transformed into a Hermite normal form and the bottom part of the transformation matrix is returned. The bottom part of the transformation matrix is used to solve the syndrome equation.
"""
overcomplete_syndrome_preperation(M::Matrix{Float64}) = overcomplete_syndrome_preperation(round.(Int64, sqrt(2) * M))


"""
    interger_solve(V::ColumnHermite, b::Vector)

Solves the syndrome equation using the Hermite normal form of the syndrome matrix and the bottom part of the transformation matrix.
"""
function interger_solve(V::ColumnHermite, b::Vector)
    B, U = V
    B = B[:, 1:size(B, 1)]
    c = zeros(size(U, 1))
    c[1:size(b, 1)] = inv(B) * b

    return U * c
end


"""
    integer_solve(V::BottomSystem, s::Vector)

Solves the syndrome equation using the bottom part of the transformation matrix and the syndrome.
"""
integer_solve(V::BottomSystem, s::Vector) = interger_solve(V.VbH, -V.Vb * s)





"""
    compute_syndrome(M::AbstractMatrix, J::AbstractMatrix, error::Vector{Float64})

Computes the syndrome of a given error using the symplectic form J.
"""
function compute_syndrome(M::AbstractMatrix, J::AbstractMatrix, error::Vector{Float64})
    return M * J * error .% 1
end


function compute_eta(M::AbstractMatrix, syndrome::Vector{Float64})
    return inv(M * symplectic_form(size(M)[2])) * syndrome
end


function compute_eta_overcomplete(Mhr::AbstractMatrix, Vt::AbstractMatrix, syndrome::Vector{Float64}, integer_solution::AbstractVector)
    return inv(Mhr * symplectic_form(Int64(size(Mhr)[1]/2))) * Vt* (syndrome + integer_solution)
end


# rep_code = Int64[1 1 0; 0 1 1]
# Id = Matrix{Int64}(2I, 3, 3)
# M = Matrix([rep_code' Id']') / sqrt(2)


# n = minimum(size(M))
# Mh, Mr, V = overcomplete_syndrome_preperation(M)

# error = randn(n)
# s = M * error .% 1

# integer_solve(V, s)


# d = 3

# H = zeros(Float64, d - 1, d)
# for i in 1:(d-1)
#     H[i, i] = 1
#     H[i, i+1] = 1

# end

# H

# bit_flip = false

# if bit_flip
#     M_H = [H / sqrt(2) zeros(Float64, d - 1, d)]
# else
#     M_H = [zeros(Float64, d - 1, d) H / sqrt(2)]
# end


# M_H
# GKP_generators = Matrix{Float64}(2 * I, 2 * d, 2 * d)

# M = [M_H; GKP_generators]