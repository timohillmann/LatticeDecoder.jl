using LinearAlgebra
using NormalForms

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
    compute_syndrome(M::AbstractMatrix, J, error::Vector{Float64})

Computes the syndrome of a given error using the symplectic form J.
"""
function compute_syndrome(M::AbstractMatrix, J, error::Vector{Float64})
    return M * J * error .% 1
end


"""
    compute_syndrome(M::AbstractMatrix, error::Vector{Float64})

Computes the syndrome of a given error using the standard symplectic form.
"""
function compute_syndrome(M::AbstractMatrix, error::Vector{Float64})
    return M * error .% 1

end


function compute_eta()
    return M * J * error
end

