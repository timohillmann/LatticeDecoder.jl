using LatticeAlgorithms
using SparseArrays: sparse, SparseMatrixCSC

function qpqp_to_qqpp(M::AbstractMatrix)
    # move columns with even indices to the right
    M_l = M[:, collect(1:2:size(M, 2))]
    M_r = M[:, collect(2:2:size(M, 2))]

    M_new = [M_l M_r]
    # move rows with even indices to the bottom
    M_t = M_new[collect(1:2:size(M_new, 1)), :]
    M_b = M_new[collect(2:2:size(M_new, 1)), :]
    M_new = [M_t; M_b]
end

function surface_code_logicals(d::Int)
    X = surface_code_X_logicals(d)
    Z = surface_code_Z_logicals(d)

    tot_logicals = length(X) + length(Z)

    L = zeros(Float64, tot_logicals, 2 * d^2)
    for i in eachindex(X)
        L[i, X[i]] .= 1.0
    end
    for i in eachindex(Z)
        L[length(X)+i, d^2 .+ Z[i]] .= 1.0
    end

    return L / sqrt(2)
end

function balance_weights!(mat::SparseMatrixCSC{Int64, Int64})
    for i in 1:size(mat, 1)
        row = mat[i, :]
        num_nzvals = count(!iszero, row)
        if num_nzvals == 1
            nz_idx = row.nzind[1]
            # find the row that has a 1 in the same column
            for j in 1:size(mat, 1)
                if j == i
                    continue
                end
                if mat[j, nz_idx] == 1
                    row2 = mat[j, :]
                    mat[i, :] .= row2 .- row
                    break
                end
            end
        end
    end
end



function GKP_Surface_Code(d::Int, balance_hamming_weight::Bool=true)
    H = qpqp_to_qqpp(surface_code_M(d))
    L = surface_code_logicals(d)

    if balance_hamming_weight
        H_CSC = sparse(H)
        H_CSC = round.(Int64, sqrt(2) * H_CSC)
        balance_weights!(H_CSC)
        H = Matrix(H_CSC / sqrt(2))
    end

    return QuantumCode(H, L)
end