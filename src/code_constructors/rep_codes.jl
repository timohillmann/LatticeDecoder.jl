

struct QuantumCode
    code::AbstractMatrix{Float64}
    logical::AbstractMatrix{Float64}
    J::AbstractMatrix{Float64}
    QuantumCode(code::AbstractMatrix{Float64}, logical::AbstractMatrix{Float64}) = new(code, logical, symplectic_form(size(code, 2)))
end

function is_logical_error(code::QuantumCode, error::Vector{Float64})
    # check if symplectic product is integer
    return all(code.logical * code.J * error .% 1 .== 0)
end


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
function GKP_Rep_Code(d::Int, bit_flip=false)
    return QuantumCode(gkp_rep_code(d, bit_flip), rep_code_logical(d, bit_flip))
end