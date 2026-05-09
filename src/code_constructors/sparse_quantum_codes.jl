using LinearAlgebra
using NPZ
using Random
using SparseArrays

const DEFAULT_SPARSE_CODE_MAX_ITERS = 500_000

"""
    load_sparse_quantum_code(path; hx_key=:Hx, hz_key=:Hz, balance_weights=false)

Load CSS quantum-code check matrices from an `.npz` file. Dense arrays are read
directly from `Hx`/`Hz` or `hx`/`hz`. SciPy sparse matrices are reconstructed
from prefixed `data`, `indices`, `indptr`, `shape`, and `format` fields.
When `balance_weights=true`, weight-one rows are replaced by a nearby
same-column row difference before returning the matrices.
"""
function load_sparse_quantum_code(
    path::AbstractString;
    hx_key::Union{Symbol,String}=:Hx,
    hz_key::Union{Symbol,String}=:Hz,
    balance_weights::Bool=false,
)
    data = _read_npz(path)
    Hx = _load_matrix_from_npz(data, hx_key, (:Hx, :hx))
    Hz = _load_matrix_from_npz(data, hz_key, (:Hz, :hz))
    if balance_weights
        Hx = _balanced_weight_matrix(Hx)
        Hz = _balanced_weight_matrix(Hz)
    end
    return (Hx=Hx, Hz=Hz, data=data)
end

"""
    enlarge_css_generators(Hx, Hz; p=2, method=:heuristic, max_iters=500_000, rng=Random.default_rng())

Enlarge CSS check matrices into q- and p-sector lattice generators. The
default `:heuristic` method selects independent rows and appends GKP
stabilizers to produce square full-rank lattice generators. `:trivial` returns
the redundant generators `[H; pI]`.
"""
function enlarge_css_generators(
    Hx,
    Hz;
    p::Integer=2,
    method::Symbol=:heuristic,
    max_iters::Integer=DEFAULT_SPARSE_CODE_MAX_ITERS,
    rng::AbstractRNG=Random.default_rng(),
)
    Hx_int, Hz_int = _validate_css_inputs(Hx, Hz, p)

    if method === :heuristic
        return (
            Mqq=_heuristic_enlarge_generator(Hx_int, Int(p); max_iters=Int(max_iters), rng=rng),
            Mpp=_heuristic_enlarge_generator(Hz_int, Int(p); max_iters=Int(max_iters), rng=rng),
        )
    elseif method === :trivial
        return (
            Mqq=_trivial_enlarge_generator(Hx_int, Int(p)),
            Mpp=_trivial_enlarge_generator(Hz_int, Int(p)),
        )
    else
        throw(ArgumentError("unsupported enlargement method $(repr(method)); expected :heuristic or :trivial"))
    end
end

"""
    write_enlarged_sparse_quantum_code(input_path, output_path; kwargs...)

Load `input_path`, enlarge `Hx` and `Hz`, and write `output_path` with new
`Mqq` and `Mpp` keys while preserving the original `.npz` contents.
"""
function write_enlarged_sparse_quantum_code(
    input_path::AbstractString,
    output_path::AbstractString;
    hx_key::Union{Symbol,String}=:Hx,
    hz_key::Union{Symbol,String}=:Hz,
    p::Integer=2,
    method::Symbol=:heuristic,
    max_iters::Integer=DEFAULT_SPARSE_CODE_MAX_ITERS,
    rng::AbstractRNG=Random.default_rng(),
)
    loaded = load_sparse_quantum_code(input_path; hx_key=hx_key, hz_key=hz_key)
    enlarged = enlarge_css_generators(
        loaded.Hx,
        loaded.Hz;
        p=p,
        method=method,
        max_iters=max_iters,
        rng=rng,
    )

    output = Dict{String,Any}(loaded.data)
    output["Mqq"] = enlarged.Mqq
    output["Mpp"] = enlarged.Mpp
    npzwrite(output_path, output)
    return enlarged
end

function balance_weights!(mat::SparseMatrixCSC{T,I}) where {T<:Integer,I<:Integer}
    for i in 1:size(mat, 1)
        row = mat[i, :]
        if length(row.nzind) == 1
            nz_idx = row.nzind[1]
            for j in 1:size(mat, 1)
                j == i && continue
                if mat[j, nz_idx] == one(T)
                    row2 = mat[j, :]
                    mat[i, :] .= row2 .- row
                    break
                end
            end
        end
    end
    return mat
end

function modp_rank(M, p::Integer)
    p_int = _validate_prime_modulus(p)
    A = mod.(_integer_matrix(M), p_int)
    m, n = size(A)
    rank = 0

    for col in 1:n
        pivot = 0
        for row in (rank + 1):m
            if !iszero(A[row, col])
                pivot = row
                break
            end
        end
        iszero(pivot) && continue

        rank += 1
        if pivot != rank
            A[rank, :], A[pivot, :] = A[pivot, :], A[rank, :]
        end

        inv_pivot = invmod(A[rank, col], p_int)
        A[rank, :] .= mod.(A[rank, :] .* inv_pivot, p_int)

        for row in 1:m
            row == rank && continue
            factor = A[row, col]
            if !iszero(factor)
                A[row, :] .= mod.(A[row, :] .- factor .* A[rank, :], p_int)
            end
        end
    end

    return rank
end

function _read_npz(path::AbstractString)
    try
        data = npzread(path)
        data isa AbstractDict || throw(ArgumentError("expected an .npz archive containing named arrays"))
        return data
    catch err
        msg = sprint(showerror, err)
        throw(ArgumentError("could not read $(path) as a supported .npz archive; pickled Python object arrays are not supported. Original error: $(msg)"))
    end
end

function _load_matrix_from_npz(data::AbstractDict, preferred_key, fallback_keys)
    candidates = String[string(preferred_key)]
    append!(candidates, string.(fallback_keys))
    unique!(candidates)

    for key in candidates
        if haskey(data, key)
            return _integer_matrix(data[key])
        end
    end

    for key in candidates
        sparse_matrix = _maybe_load_scipy_sparse(data, key)
        sparse_matrix === nothing || return sparse_matrix
    end

    throw(KeyError("could not find matrix $(repr(string(preferred_key))) or sparse fields for candidates $(candidates)"))
end

function _maybe_load_scipy_sparse(data::AbstractDict, name::AbstractString)
    data_key = _find_sparse_field(data, name, "data")
    indices_key = _find_sparse_field(data, name, "indices")
    indptr_key = _find_sparse_field(data, name, "indptr")
    shape_key = _find_sparse_field(data, name, "shape")
    format_key = _find_sparse_field(data, name, "format")

    required = (data_key, indices_key, indptr_key, shape_key, format_key)
    all(!isnothing, required) || return nothing

    values = _integer_vector(data[data_key])
    indices = Int.(vec(data[indices_key])) .+ 1
    indptr = Int.(vec(data[indptr_key])) .+ 1
    shape = Int.(vec(data[shape_key]))
    length(shape) == 2 || throw(ArgumentError("sparse matrix $(name) has invalid shape field"))
    m, n = shape
    fmt = _decode_sparse_format(data[format_key])

    if fmt == "csr"
        rows = Vector{Int}(undef, length(values))
        cols = Vector{Int}(undef, length(values))
        cursor = 1
        for row in 1:m
            for ptr in indptr[row]:(indptr[row + 1] - 1)
                rows[cursor] = row
                cols[cursor] = indices[ptr]
                cursor += 1
            end
        end
        return sparse(rows, cols, values, m, n)
    elseif fmt == "csc"
        rows = Vector{Int}(undef, length(values))
        cols = Vector{Int}(undef, length(values))
        cursor = 1
        for col in 1:n
            for ptr in indptr[col]:(indptr[col + 1] - 1)
                rows[cursor] = indices[ptr]
                cols[cursor] = col
                cursor += 1
            end
        end
        return sparse(rows, cols, values, m, n)
    else
        throw(ArgumentError("unsupported SciPy sparse format $(repr(fmt)) for matrix $(name); expected csr or csc"))
    end
end

function _find_sparse_field(data::AbstractDict, name::AbstractString, field::AbstractString)
    names = unique(String[
        "$(name)/$(field)",
        "$(name)_$(field)",
        "$(name).$(field)",
        "$(lowercase(name))/$(field)",
        "$(lowercase(name))_$(field)",
        "$(lowercase(name)).$(field)",
    ])
    for key in names
        haskey(data, key) && return key
    end
    return nothing
end

function _decode_sparse_format(value)
    if value isa AbstractString
        return lowercase(String(value))
    elseif value isa AbstractArray{UInt8}
        return lowercase(String(vec(value)))
    elseif value isa AbstractArray
        flat = vec(value)
        if eltype(flat) <: UInt8
            return lowercase(String(flat))
        elseif length(flat) == 1
            return lowercase(String(flat[1]))
        end
    end
    return lowercase(String(value))
end

function _validate_css_inputs(Hx, Hz, p::Integer)
    p_int = _validate_prime_modulus(p)
    Hx_int = _integer_matrix(Hx)
    Hz_int = _integer_matrix(Hz)
    size(Hx_int, 2) == size(Hz_int, 2) ||
        throw(DimensionMismatch("Hx and Hz must have the same number of columns; got $(size(Hx_int, 2)) and $(size(Hz_int, 2))"))
    modp_rank(Hx_int, p_int) > 0 || throw(ArgumentError("Hx has rank 0 modulo $(p_int)"))
    modp_rank(Hz_int, p_int) > 0 || throw(ArgumentError("Hz has rank 0 modulo $(p_int)"))
    return Hx_int, Hz_int
end

function _balanced_weight_matrix(M)
    M_int = _integer_matrix(M)
    S = sparse(M_int)
    balance_weights!(S)
    return M_int isa AbstractSparseMatrix ? S : Matrix(S)
end

function _validate_prime_modulus(p::Integer)
    p_int = Int(p)
    p_int > 1 || throw(ArgumentError("p must be an integer greater than 1"))
    for d in 2:floor(Int, sqrt(p_int))
        if p_int % d == 0
            throw(ArgumentError("p must be prime so modular inverses exist; got $(p_int)"))
        end
    end
    return p_int
end

function _integer_matrix(M)
    if M isa SparseMatrixCSC
        _assert_integer_like(nonzeros(M))
        return SparseMatrixCSC{Int,Int}(M)
    elseif M isa AbstractSparseMatrix
        return sparse(_integer_matrix(Matrix(M)))
    elseif M isa AbstractMatrix
        _assert_integer_like(M)
        return Int.(round.(M))
    else
        throw(ArgumentError("expected an integer-like dense or sparse matrix, got $(typeof(M)); pickled Python object arrays are not supported"))
    end
end

function _integer_vector(v)
    _assert_integer_like(v)
    return Int.(round.(vec(v)))
end

function _assert_integer_like(values)
    for x in values
        if !(x isa Integer) && !(x isa Real && isfinite(x) && x == round(x))
            throw(ArgumentError("matrix entries must be integer-like; found $(repr(x)) of type $(typeof(x))"))
        end
    end
    return nothing
end

function _heuristic_enlarge_generator(H, p::Int; max_iters::Int, rng::AbstractRNG)
    G = Matrix(_integer_matrix(H))
    k, n = size(G)
    r = modp_rank(G, p)
    r > 0 || throw(ArgumentError("matrix has rank 0 modulo $(p)"))

    cols = _select_independent_columns(G, r, p; max_iters=max_iters, rng=rng)
    rows = _select_independent_rows(@view(G[:, cols]), r, p; max_iters=max_iters, rng=rng)

    Gprime = G[rows, :]
    unselected = setdiff(1:n, cols)
    extra = zeros(Int, n - r, n)
    for (i, col) in enumerate(unselected)
        extra[i, col] = p
    end

    return vcat(Gprime, extra)
end

function _trivial_enlarge_generator(H, p::Int)
    G = Matrix(_integer_matrix(H))
    n = size(G, 2)
    return vcat(G, p .* Matrix{Int}(I, n, n))
end

function _select_independent_columns(G::AbstractMatrix{Int}, r::Int, p::Int; max_iters::Int, rng::AbstractRNG)
    _, n = size(G)
    perm = collect(1:n)
    best = Vector{Int}()
    best_score = Inf

    for _ in 1:max_iters
        shuffle!(rng, perm)
        cols = _independent_indices((j -> @view(G[:, j])), n, r, p; order=perm)
        if length(cols) == r
            score = _gram_logdet(@view(G[:, cols]))
            if score < best_score
                best_score = score
                best = copy(cols)
            end
        end
    end

    length(best) == r || throw(ArgumentError("could not find $(r) independent columns modulo $(p) after $(max_iters) iterations"))
    return best
end

function _select_independent_rows(G::AbstractMatrix{Int}, r::Int, p::Int; max_iters::Int, rng::AbstractRNG)
    k, _ = size(G)
    perm = collect(1:k)
    best = Vector{Int}()
    best_score = Inf

    for _ in 1:max_iters
        shuffle!(rng, perm)
        rows = _independent_indices((i -> @view(G[i, :])), k, r, p; order=perm)
        if length(rows) == r
            score = _gram_logdet(transpose(@view(G[rows, :])))
            if score < best_score
                best_score = score
                best = copy(rows)
            end
        end
    end

    length(best) == r || throw(ArgumentError("could not find $(r) independent rows modulo $(p) after $(max_iters) iterations"))
    return best
end

function _independent_indices(get_vector, count::Int, target_rank::Int, p::Int; order=1:count)
    basis = Vector{Vector{Int}}()
    pivots = Int[]
    selected = Int[]

    for idx in order
        v = mod.(Int.(get_vector(idx)), p)
        if _insert_modp_basis!(basis, pivots, v, p)
            push!(selected, idx)
            length(selected) == target_rank && break
        end
    end

    return selected
end

function _insert_modp_basis!(basis::Vector{Vector{Int}}, pivots::Vector{Int}, v::Vector{Int}, p::Int)
    w = copy(v)
    for (basis_vector, pivot) in zip(basis, pivots)
        factor = w[pivot]
        if !iszero(factor)
            w .= mod.(w .- factor .* basis_vector, p)
        end
    end

    pivot = findfirst(!iszero, w)
    pivot === nothing && return false

    inv_pivot = invmod(w[pivot], p)
    w .= mod.(w .* inv_pivot, p)
    push!(basis, w)
    push!(pivots, pivot)
    return true
end

function _gram_logdet(A)
    gram = Matrix(transpose(A) * A)
    value, sign = logabsdet(float.(gram))
    sign > 0 || return Inf
    return value
end
