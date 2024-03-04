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

    Vt = U[1:size(Mh, 2), :]
    Vb = U[size(Mh, 2)+1:end, :]
    VbH = hnfc(Vb)
    return BottomSystem(Vb, VbH)
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



rep_code = Int64[1 1 0; 0 1 1]
Id = Matrix{Int64}(2I, 3, 3)
M = Matrix([rep_code' Id']') / sqrt(2)


n = minimum(size(M))
V = overcomplete_syndrome_preperation(M)

error = randn(n)
s = M * error .% 1

integer_solve(V, s)