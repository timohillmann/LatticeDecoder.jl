import LatticeAlgorithms


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
    X =  LatticeAlgorithms.surface_code_X_logicals(d)
    Z =  LatticeAlgorithms.surface_code_Z_logicals(d)

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

function GKP_Surface_Code(d::Int)
    H = qpqp_to_qqpp(LatticeAlgorithms.surface_code_M(d))
    L = surface_code_logicals(d)
    return QuantumCode(H, L)
end