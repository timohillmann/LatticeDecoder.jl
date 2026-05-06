using DelimitedFiles

function load_txt_matrix(file::String)
    return readdlm(file, String)
end

path = "/Users/timo/Documents/GitHub/LatticeDecoder.jl/nithin_codes_N_544_K_80_L_16_hx_lattice.txt"

hx = load_txt_matrix(path);


function convert_to_vec(vec::SubString)
    return [parse(Int, x) for x in split(vec, "")]
end


new_hx = []
for i in 1:size(hx, 1)
    # println(i, " ", hx[i, 1])
    push!(new_hx, [parse(Int, x) for x in split(hx[i, 1], "")])
end

new_hx = hcat(new_hx...)';

using LinearAlgebraX

# detx(new_hx)

# find row weights of new_hx
using SparseArrays

H = sparse(new_hx)


function balance_weights(mat::SparseMatrixCSC)
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

            # subtract row from row2 and write it to mat[i, :]
        end
    end
end
using NPZ

balance_weights(H);

H = Matrix(H)

npzwrite("/Users/timo/Documents/GitHub/LatticeDecoder.jl/data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_hx_balanced.npy", H)


G = invx(H)
G = Float64.(G)
using NPZ

npzwrite("/Users/timo/Documents/GitHub/LatticeDecoder.jl/data/generator_matrices/binary_codes/nithin_codes_N_544_K_80_L_16_gx_balanced.npy", G)