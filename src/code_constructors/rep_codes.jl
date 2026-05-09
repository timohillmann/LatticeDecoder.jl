

ACCEPTANCE_THRESHOLD = 1E-5

struct QuantumCode
    code::AbstractMatrix{Float64}
    logical::AbstractMatrix{Float64}
    J::AbstractMatrix{Float64}
    QuantumCode(code::AbstractMatrix{Float64}, logical::AbstractMatrix{Float64}) = new(code, logical, symplectic_form(Int64(size(code)[2] / 2)))
end



"""
    gkp_rep_code(d::Int, bit_flip=false)

Construct the GKP-Repetition code with `d` modes. If `bit_flip` is true, then the protected qubit is encoded in the position basis, otherwise it is encoded in the momentum basis.
"""
function gkp_rep_code(d::Int, bit_flip=false, reduced=false)
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
    if reduced
        return Float64.(stack_gkp_generator(matrix(Z2, Int64.(round.(M_H))))) / sqrt(2)
    else
        GKP_generators = Matrix{Float64}(2 * I, 2 * d, 2 * d)
        return [GKP_generators; M_H] / sqrt(2)
    end

end


"""
    rep_code_logical(d::Int, bit_flip=false)

Construct the protected logical operator for the GKP-Repetition code with `d` modes. If `bit_flip` is true, then the protected qubit is encoded in the position basis, otherwise it is encoded in the momentum basis.
"""
function rep_code_logical(d::Int, bit_flip=false)
    L = zeros(Float64, 1, 2 * d)

    if bit_flip
        L[1, (d+1):end] .= 1
    else
        L[1, 1:d] .= 1
    end

    return L / sqrt(2)
end


"""
    GKP_Rep_Code(d::Int, bit_flip=false)

Construct the GKP-Repetition code QuantumCode object with `d` modes. If `bit_flip` is true, then the protected qubit is encoded in the position basis, otherwise it is encoded in the momentum basis.
"""
function GKP_Rep_Code(d::Int, bit_flip=false, reduced=false)
    return QuantumCode(gkp_rep_code(d, bit_flip, reduced), rep_code_logical(d, bit_flip))
end