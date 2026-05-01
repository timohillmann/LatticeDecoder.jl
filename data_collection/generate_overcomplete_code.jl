using NormalForms
using LLLplus
using LatticeDecoder
using LinearAlgebra
using NPZ
using SparseArrays: sparse, SparseMatrixCSC
using IterTools: product



function load_code(code_name::String)

    p = parse(Int, code_name[end])
    code_data = npzread("/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name).npz")
    hx = code_data["hx"];
    n = size(hx, 2);
    Id_block = p  * I(n);
    H = vcat(hx, Id_block) ;
    return H, code_data["lz"]

end

function generate_code_data(code_name::String)
    


    q = parse(Int, code_name[end])
    # Build the code
    H, lz = load_code(code_name)
    println("Code loaded.")
    flush(stdout)

    H = BigInt.(H)
    n = size(H, 2)
    H_R, _ = hnfr(H)
    println("Code reduced.")
    flush(stdout)
    H_R = H_R[1:n, :]  # drop zero rows
    G = inv(H_R / sqrt(q))
    println("Generator initialized.")
    flush(stdout)
    Gp, _ = lll(G')
    println("Generator reduced.")
    flush(stdout)
    G = Float64.(Matrix(Gp'))

    PATH = "/Users/timo/Documents/LatticeDecoder.jl/data/generator_matrices/bivariate_bicycle/$(code_name)_overcomplete.npz";
    code = npzwrite(PATH, Dict("hx" => Int.(H), "Gz" => G, "lz" => lz));
end


function main()
    codes = ["128_2_8_p5", "176_2_12_p5"];
    for code in codes
        generate_code_data(code)
    end
end

main()