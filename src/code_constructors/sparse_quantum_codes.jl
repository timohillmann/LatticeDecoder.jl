using LinearAlgebra
using NPZ
using Random
using SparseArrays

raw"""
    construction_a_css_gkp(Hx, Hz; p=2, l=1)

Construct the sparse, overcomplete Construction-A GKP generator associated
with CSS checks over qudits of dimension `q = p^l`. `p` must be prime and
`l >= 1`. In `qqpp` coordinate order the returned generator is

```math
M = q^{-1/2}\begin{pmatrix}
H_X & 0 \\
0 & H_Z \\
qI_n & 0 \\
0 & qI_n
\end{pmatrix}.
```

The CSS condition `Hx * transpose(Hz) == 0 (mod q)` is checked exactly. The
result contains sparse full and sector generators in both integer and scaled
form. `Hx_overcomplete` and `Hz_overcomplete` are the literal `[H; qI]`
matrices. The `*_balanced` variants replace each covered `q e_i` row by
`h_j - q e_i`, using a minimum-weight row `h_j` incident on coordinate `i`.
This is a unimodular change of the redundant rows, so it preserves the lattice
while removing degree-one checks. These sector matrices are intended for CSS
LDLC decoding; a square completion of the same sector lattice is still needed
for hard decisions.
"""
function construction_a_css_gkp(
    Hx,
    Hz;
    p::Integer=2,
    l::Integer=1,
)
    p_int = _validate_prime_modulus(p)
    l_int = Int(l)
    l_int >= 1 || throw(ArgumentError("l must be a positive integer; got $(l)"))

    q_big = big(p_int)^l_int
    q_big <= typemax(Int) || throw(OverflowError("p^l does not fit in Int: $(p_int)^$(l_int)"))
    q = Int(q_big)

    Hx_int = _integer_matrix(Hx)
    Hz_int = _integer_matrix(Hz)
    size(Hx_int, 2) == size(Hz_int, 2) || throw(DimensionMismatch(
        "Hx and Hz must have the same number of columns; got $(size(Hx_int, 2)) and $(size(Hz_int, 2))",
    ))

    commutator = Hx_int * transpose(Hz_int)
    offending = _first_nondivisible_entry(commutator, q)
    offending === nothing || throw(ArgumentError(
        "CSS condition failed modulo $(q): (Hx * transpose(Hz))$(offending.index) = " *
        "$(offending.value) is not divisible by $(q)",
    ))

    n = size(Hx_int, 2)
    Hx_sparse = sparse(Hx_int)
    Hz_sparse = sparse(Hz_int)
    zero_xz = spzeros(Int, size(Hx_int, 1), n)
    zero_zx = spzeros(Int, size(Hz_int, 1), n)
    H_integer = vcat(
        hcat(Hx_sparse, zero_xz),
        hcat(zero_zx, Hz_sparse),
    )
    gkp_integer = spdiagm(0 => fill(q, 2n))
    M_integer = vcat(H_integer, gkp_integer)

    sector_gkp_integer = spdiagm(0 => fill(q, n))
    Hx_overcomplete_integer = vcat(Hx_sparse, sector_gkp_integer)
    Hz_overcomplete_integer = vcat(Hz_sparse, sector_gkp_integer)
    Hx_balanced_integer = _balanced_construction_a_sector(Hx_sparse, q)
    Hz_balanced_integer = _balanced_construction_a_sector(Hz_sparse, q)
    scale = sqrt(Float64(q))

    return (
        M=M_integer ./ scale,
        M_integer=M_integer,
        Hx_overcomplete=Hx_overcomplete_integer ./ scale,
        Hz_overcomplete=Hz_overcomplete_integer ./ scale,
        Hx_balanced=Hx_balanced_integer ./ scale,
        Hz_balanced=Hz_balanced_integer ./ scale,
        Hx_overcomplete_integer=Hx_overcomplete_integer,
        Hz_overcomplete_integer=Hz_overcomplete_integer,
        Hx_balanced_integer=Hx_balanced_integer,
        Hz_balanced_integer=Hz_balanced_integer,
        modulus=q,
        prime=p_int,
        exponent=l_int,
        scale=scale,
    )
end

function _balanced_construction_a_sector(H::SparseMatrixCSC{Int,Int}, modulus::Int)
    m, n = size(H)
    row_columns = [Int[] for _ in 1:m]
    row_values = [Int[] for _ in 1:m]
    rows = rowvals(H)
    values = nonzeros(H)
    for col in 1:n
        for ptr in nzrange(H, col)
            row = rows[ptr]
            push!(row_columns[row], col)
            push!(row_values[row], values[ptr])
        end
    end

    bottom_rows = Int[]
    bottom_columns = Int[]
    bottom_values = Int[]
    for col in 1:n
        source = 0
        source_weight = typemax(Int)
        for ptr in nzrange(H, col)
            candidate = rows[ptr]
            candidate_weight = length(row_columns[candidate])
            if candidate_weight < source_weight
                source = candidate
                source_weight = candidate_weight
            end
        end

        if iszero(source)
            push!(bottom_rows, col)
            push!(bottom_columns, col)
            push!(bottom_values, modulus)
            continue
        end

        for (source_col, value) in zip(row_columns[source], row_values[source])
            push!(bottom_rows, col)
            push!(bottom_columns, source_col)
            push!(bottom_values, value)
        end
        push!(bottom_rows, col)
        push!(bottom_columns, col)
        push!(bottom_values, -modulus)
    end

    bottom = sparse(bottom_rows, bottom_columns, bottom_values, n, n)
    dropzeros!(bottom)
    return vcat(H, bottom)
end

function _first_nondivisible_entry(M, modulus::Int)
    if M isa SparseMatrixCSC
        rows = rowvals(M)
        for col in 1:size(M, 2)
            for ptr in nzrange(M, col)
                value = nonzeros(M)[ptr]
                if !iszero(mod(value, modulus))
                    return (index=(rows[ptr], col), value=value)
                end
            end
        end
    else
        for index in CartesianIndices(M)
            value = M[index]
            if !iszero(mod(value, modulus))
                return (index=Tuple(index), value=value)
            end
        end
    end
    return nothing
end

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
    enlarge_css_generators(Hx, Hz; p=2, method=:heuristic, max_iters=LatticeDecoder.DEFAULT_SPARSE_CODE_MAX_ITERS, rng=Random.default_rng(), progress_io=stdout, progress_every=1000, score_every=1)

Enlarge CSS check matrices into q- and p-sector lattice generators. The
`:systematic` method constructs a mod-`p` row-echelon basis and appends GKP
stabilizers, giving determinant exactly `p^(n-r)` for each sector. The
`:sparse_echelon` method does sparse mod-2 elimination with low-fill pivots,
giving the same exact determinant volume while keeping rows much sparser than
full reduction. The `:heuristic` method jointly searches for a unimodular
full-rank minor, retains the corresponding original rows, and appends GKP
stabilizers. It throws rather than returning a lattice with the wrong volume if
no determinant-`±1` minor is found. When there are at most two redundant rows,
all row removals are checked for each candidate column set. Progress is written
to `progress_io` every `progress_every` iterations, and candidates are scored
every `score_every` iterations. Use `progress_io=nothing` to disable progress.
`:trivial` returns the redundant generators `[H; pI]`.
"""
function enlarge_css_generators(
    Hx,
    Hz;
    p::Integer=2,
    method::Symbol=:heuristic,
    max_iters::Integer=LatticeDecoder.DEFAULT_SPARSE_CODE_MAX_ITERS,
    rng::AbstractRNG=Random.default_rng(),
    progress_io::Union{Nothing,IO}=stdout,
    progress_every::Integer=1_000,
    score_every::Integer=1,
)
    Hx_int, Hz_int = _validate_css_inputs(Hx, Hz, p)
    progress_every_int = _validate_progress_every(progress_every)
    score_every_int = _validate_progress_every(score_every)

    if method === :systematic
        return (
            Mqq=_systematic_enlarge_generator(Hx_int, Int(p)),
            Mpp=_systematic_enlarge_generator(Hz_int, Int(p)),
        )
    elseif method === :sparse_echelon
        return (
            Mqq=_sparse_echelon_enlarge_generator(
                Hx_int,
                Int(p);
                progress_io=progress_io,
                progress_every=progress_every_int,
                label="Mqq",
            ),
            Mpp=_sparse_echelon_enlarge_generator(
                Hz_int,
                Int(p);
                progress_io=progress_io,
                progress_every=progress_every_int,
                label="Mpp",
            ),
        )
    elseif method === :heuristic
        return (
            Mqq=_heuristic_enlarge_generator(
                Hx_int,
                Int(p);
                max_iters=Int(max_iters),
                rng=rng,
                progress_io=progress_io,
                progress_every=progress_every_int,
                score_every=score_every_int,
                label="Mqq",
            ),
            Mpp=_heuristic_enlarge_generator(
                Hz_int,
                Int(p);
                max_iters=Int(max_iters),
                rng=rng,
                progress_io=progress_io,
                progress_every=progress_every_int,
                score_every=score_every_int,
                label="Mpp",
            ),
        )
    elseif method === :trivial
        return (
            Mqq=_trivial_enlarge_generator(Hx_int, Int(p)),
            Mpp=_trivial_enlarge_generator(Hz_int, Int(p)),
        )
    else
        throw(ArgumentError("unsupported enlargement method $(repr(method)); expected :systematic, :sparse_echelon, :heuristic, or :trivial"))
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
    max_iters::Integer=LatticeDecoder.DEFAULT_SPARSE_CODE_MAX_ITERS,
    rng::AbstractRNG=Random.default_rng(),
    progress_io::Union{Nothing,IO}=stdout,
    progress_every::Integer=1_000,
    score_every::Integer=1,
)
    loaded = load_sparse_quantum_code(input_path; hx_key=hx_key, hz_key=hz_key)
    enlarged = enlarge_css_generators(
        loaded.Hx,
        loaded.Hz;
        p=p,
        method=method,
        max_iters=max_iters,
        rng=rng,
        progress_io=progress_io,
        progress_every=progress_every,
        score_every=score_every,
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
    _has_nonzero_modp_entry(Hx_int, p_int) || throw(ArgumentError("Hx has rank 0 modulo $(p_int)"))
    _has_nonzero_modp_entry(Hz_int, p_int) || throw(ArgumentError("Hz has rank 0 modulo $(p_int)"))
    return Hx_int, Hz_int
end

function _has_nonzero_modp_entry(M, p::Int)
    values = M isa SparseMatrixCSC ? nonzeros(M) : M
    for value in values
        !iszero(mod(value, p)) && return true
    end
    return false
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

function _validate_progress_every(progress_every::Integer)
    progress_every_int = Int(progress_every)
    progress_every_int > 0 ||
        throw(ArgumentError("progress_every must be a positive integer; got $(progress_every)"))
    return progress_every_int
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

function _heuristic_enlarge_generator(
    H,
    p::Int;
    max_iters::Int,
    rng::AbstractRNG,
    progress_io::Union{Nothing,IO},
    progress_every::Int,
    score_every::Int,
    label::AbstractString,
)
    G = Matrix(_integer_matrix(H))
    k, n = size(G)
    r = modp_rank(G, p)
    r > 0 || throw(ArgumentError("matrix has rank 0 modulo $(p)"))

    rows, cols = _select_unimodular_submatrix(
        G,
        r,
        p;
        max_iters=max_iters,
        rng=rng,
        progress_io=progress_io,
        progress_every=progress_every,
        score_every=score_every,
        label=label,
    )

    Gprime = G[rows, :]
    unselected = setdiff(1:n, cols)
    extra = zeros(Int, n - r, n)
    for (i, col) in enumerate(unselected)
        extra[i, col] = p
    end

    return vcat(Gprime, extra)
end

function _select_unimodular_submatrix(
    G::AbstractMatrix{Int},
    r::Int,
    p::Int;
    max_iters::Int,
    rng::AbstractRNG,
    progress_io::Union{Nothing,IO},
    progress_every::Int,
    score_every::Int,
    label::AbstractString,
)
    k, n = size(G)
    column_order = collect(1:n)
    row_order = collect(1:k)
    best_rows = Int[]
    best_cols = Int[]
    best_logabsdet = Inf
    start_time = time()

    for iter in 1:max_iters
        shuffle!(rng, column_order)
        cols = _independent_indices((j -> @view(G[:, j])), n, r, p; order=column_order)
        length(cols) == r || continue

        if _should_score_candidate(iter, max_iters, score_every)
            shuffle!(rng, row_order)
            rows, logabsdet_value = _best_rows_for_columns(G, cols, r, p, row_order)
            if length(rows) == r && logabsdet_value < best_logabsdet
                best_logabsdet = logabsdet_value
                best_rows = copy(rows)
                best_cols = copy(cols)
            end

            # G[rows, cols] is an integer matrix. A nonsingular minor with
            # |det| numerically equal to one is therefore unimodular.
            if length(rows) == r && isapprox(logabsdet_value, 0.0; atol=1e-8, rtol=0.0)
                _report_selection_progress(
                    progress_io,
                    "$(label) joint minor: found |det|=1 at iteration $(iter) elapsed=$(round(time() - start_time; digits=1))s";
                    final=true,
                )
                return rows, cols
            end
        end

        if _should_report_progress(iter, max_iters, progress_every)
            _report_selection_progress(
                progress_io,
                _unimodular_selection_progress_message(
                    label,
                    iter,
                    max_iters,
                    start_time,
                    best_logabsdet,
                );
                final=iter == max_iters,
            )
        end
    end

    best_det = isfinite(best_logabsdet) ? exp(best_logabsdet) : Inf
    throw(ArgumentError(
        "could not find a unimodular $(r)×$(r) minor for $(label) after $(max_iters) iterations; " *
        "best |det|≈$(round(best_det; digits=6)). Increase max_iters or use :sparse_echelon.",
    ))
end

function _best_rows_for_columns(
    G::AbstractMatrix{Int},
    cols::Vector{Int},
    r::Int,
    p::Int,
    row_order::Vector{Int},
)
    k = size(G, 1)
    base_rows = _independent_indices((i -> @view(G[i, cols])), k, r, p; order=row_order)
    length(base_rows) == r || return Int[], Inf

    # For the Kasai instances k-r=2. Express every row in one selected row
    # basis. Replacing one or two basis rows then changes the determinant by a
    # 1×1 or 2×2 complementary minor, allowing exhaustive row selection after
    # a single factorization.
    redundant = k - r
    if redundant <= 2
        A = Float64.(G[base_rows, cols])
        logdet_A, sign_A = logabsdet(A)
        iszero(sign_A) && return Int[], Inf
        coordinates = Float64.(G[:, cols]) / A
        extra_rows = setdiff(1:k, base_rows)
        absdet_A = exp(logdet_A)
        best_absdet = absdet_A
        best_rows = copy(base_rows)

        for extra in extra_rows, i in 1:r
            candidate = _rounded_integer_determinant(absdet_A * abs(coordinates[extra, i]))
            if candidate > 0 && candidate < best_absdet
                best_absdet = candidate
                best_rows = copy(base_rows)
                best_rows[i] = extra
            end
        end

        if redundant == 2
            extra1, extra2 = extra_rows
            for i in 1:(r - 1), j in (i + 1):r
                complementary_det =
                    coordinates[extra1, i] * coordinates[extra2, j] -
                    coordinates[extra1, j] * coordinates[extra2, i]
                candidate = _rounded_integer_determinant(absdet_A * abs(complementary_det))
                if candidate > 0 && candidate < best_absdet
                    best_absdet = candidate
                    best_rows = copy(base_rows)
                    best_rows[i] = extra1
                    best_rows[j] = extra2
                end
            end
        end

        # Every candidate determinant is integral; remove harmless floating
        # error before comparing it with one.
        rounded_absdet = round(best_absdet)
        isapprox(best_absdet, rounded_absdet; rtol=1e-7, atol=1e-6) &&
            (best_absdet = rounded_absdet)
        return best_rows, log(best_absdet)
    end

    logdet_value, sign = logabsdet(Float64.(G[base_rows, cols]))
    return iszero(sign) ? (Int[], Inf) : (base_rows, logdet_value)
end

function _rounded_integer_determinant(value::Float64)
    # The source matrix is integral, so every nonsingular maximal minor has a
    # positive integral absolute determinant. Rounding also suppresses the
    # small nonzero residues produced for singular candidates by floating LU.
    return round(value)
end

function _trivial_enlarge_generator(H, p::Int)
    G = Matrix(_integer_matrix(H))
    n = size(G, 2)
    return vcat(G, p .* Matrix{Int}(I, n, n))
end

function _systematic_enlarge_generator(H, p::Int)
    R, pivots = _systematic_modp_row_basis(H, p)
    _, n = size(R)
    unselected = setdiff(1:n, pivots)

    extra = zeros(Int, length(unselected), n)
    for (i, col) in enumerate(unselected)
        extra[i, col] = p
    end

    return vcat(R, extra)
end

function _sparse_echelon_enlarge_generator(
    H,
    p::Int;
    progress_io::Union{Nothing,IO},
    progress_every::Int,
    label::AbstractString,
)
    p == 2 ||
        throw(ArgumentError(":sparse_echelon currently supports p=2 only; got p=$(p)"))

    R, pivots = _sparse_echelon_mod2_row_basis(
        H;
        progress_io=progress_io,
        progress_every=progress_every,
        label=label,
    )
    _, n = size(R)
    unselected = setdiff(1:n, pivots)

    extra = zeros(Int, length(unselected), n)
    for (i, col) in enumerate(unselected)
        extra[i, col] = p
    end

    return vcat(R, extra)
end

function _systematic_modp_row_basis(H, p::Int)
    p == 2 && return _systematic_mod2_row_basis(H)

    A = mod.(Matrix(_integer_matrix(H)), p)
    m, n = size(A)
    rank = 0
    pivots = Int[]

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

        inv_pivot = invmod(A[rank, col], p)
        A[rank, :] .= mod.(A[rank, :] .* inv_pivot, p)

        for row in 1:m
            row == rank && continue
            factor = A[row, col]
            if !iszero(factor)
                A[row, :] .= mod.(A[row, :] .- factor .* A[rank, :], p)
            end
        end

        push!(pivots, col)
        rank == m && break
    end

    rank > 0 || throw(ArgumentError("matrix has rank 0 modulo $(p)"))
    return A[1:rank, :], pivots
end

function _systematic_mod2_row_basis(H)
    H_int = _integer_matrix(H)
    m, n = size(H_int)
    chunks = cld(n, 64)
    A = zeros(UInt64, m, chunks)
    _pack_mod2_rows!(A, H_int)

    rank = 0
    pivots = Int[]

    for col in 1:n
        chunk = ((col - 1) >>> 6) + 1
        mask = UInt64(1) << ((col - 1) & 63)
        pivot = 0

        for row in (rank + 1):m
            if !iszero(A[row, chunk] & mask)
                pivot = row
                break
            end
        end
        iszero(pivot) && continue

        rank += 1
        if pivot != rank
            for c in 1:chunks
                A[rank, c], A[pivot, c] = A[pivot, c], A[rank, c]
            end
        end

        for row in 1:m
            if row != rank && !iszero(A[row, chunk] & mask)
                for c in chunk:chunks
                    A[row, c] ⊻= A[rank, c]
                end
            end
        end

        push!(pivots, col)
        rank == m && break
    end

    rank > 0 || throw(ArgumentError("matrix has rank 0 modulo 2"))
    return _unpack_mod2_rows(A, rank, n), pivots
end

function _sparse_echelon_mod2_row_basis(
    H;
    progress_io::Union{Nothing,IO},
    progress_every::Int,
    label::AbstractString,
)
    H_int = _integer_matrix(H)
    m, n = size(H_int)
    rows = _mod2_row_bitsets(H_int)
    active = BitSet(i for i in 1:m if !isempty(rows[i]))
    selected_rows = Vector{Vector{Int}}()
    pivots = Int[]
    start_time = time()
    last_pivot_weight = 0

    while !isempty(active)
        pivot = _choose_sparse_echelon_pivot(rows, active)
        pivot === nothing && break

        pivot_row, pivot_col = pivot
        pivot_bits = copy(rows[pivot_row])
        last_pivot_weight = length(pivot_bits)

        push!(selected_rows, collect(pivot_bits))
        push!(pivots, pivot_col)

        affected = Int[
            row for row in active if row != pivot_row && pivot_col in rows[row]
        ]
        for row in affected
            _xor_bitset!(rows[row], pivot_bits)
            isempty(rows[row]) && delete!(active, row)
        end
        delete!(active, pivot_row)

        step = length(selected_rows)
        if step == 1 || step % progress_every == 0
            _report_selection_progress(
                progress_io,
                _sparse_echelon_progress_message(
                    label,
                    step,
                    active,
                    rows,
                    start_time,
                    last_pivot_weight,
                ),
            )
        end
    end

    rank = length(selected_rows)
    rank > 0 || throw(ArgumentError("matrix has rank 0 modulo 2"))

    R = zeros(Int, rank, n)
    for (row_idx, cols) in enumerate(selected_rows)
        for col in cols
            R[row_idx, col] = 1
        end
    end

    _report_selection_progress(
        progress_io,
        _sparse_echelon_final_message(label, rank, R, start_time);
        final=true,
    )

    return R, pivots
end

function _mod2_row_bitsets(H::SparseMatrixCSC)
    m, _ = size(H)
    rows = [BitSet() for _ in 1:m]
    row_indices, col_indices, values = findnz(H)
    for (row, col, value) in zip(row_indices, col_indices, values)
        isodd(value) || continue
        if col in rows[row]
            delete!(rows[row], col)
        else
            push!(rows[row], col)
        end
    end
    return rows
end

function _mod2_row_bitsets(H::AbstractMatrix)
    m, n = size(H)
    rows = [BitSet() for _ in 1:m]
    for col in 1:n
        for row in 1:m
            if isodd(H[row, col])
                push!(rows[row], col)
            end
        end
    end
    return rows
end

function _choose_sparse_echelon_pivot(rows::Vector{BitSet}, active::BitSet)
    col_to_rows = Dict{Int,Vector{Int}}()
    for row in active
        for col in rows[row]
            push!(get!(col_to_rows, col, Int[]), row)
        end
    end
    isempty(col_to_rows) && return nothing

    best_row = 0
    best_col = 0
    best_score = (
        typemax(Int),
        typemax(Int),
        typemax(Int),
        typemax(Int),
        typemax(Int),
        typemax(Int),
    )

    for col in sort!(collect(keys(col_to_rows)))
        candidates = col_to_rows[col]
        degree = length(candidates)
        sort!(candidates)
        for row in candidates
            weight = length(rows[row])
            fill_delta, max_new_weight = _sparse_echelon_fill_cost(rows, candidates, row)
            score = (fill_delta, weight, max_new_weight, degree, col, row)
            if score < best_score
                best_score = score
                best_row = row
                best_col = col
            end
        end
    end

    return (best_row, best_col)
end

function _sparse_echelon_fill_cost(rows::Vector{BitSet}, candidates::Vector{Int}, pivot_row::Int)
    pivot = rows[pivot_row]
    fill_delta = 0
    max_new_weight = 0
    for row in candidates
        row == pivot_row && continue
        new_weight = _symdiff_length(rows[row], pivot)
        fill_delta += new_weight - length(rows[row])
        max_new_weight = max(max_new_weight, new_weight)
    end
    return fill_delta, max_new_weight
end

function _symdiff_length(a::BitSet, b::BitSet)
    common = 0
    if length(a) <= length(b)
        for col in a
            common += col in b
        end
    else
        for col in b
            common += col in a
        end
    end
    return length(a) + length(b) - 2 * common
end

function _xor_bitset!(target::BitSet, pivot::BitSet)
    for col in pivot
        if col in target
            delete!(target, col)
        else
            push!(target, col)
        end
    end
    return target
end

function _pack_mod2_rows!(A::Matrix{UInt64}, H::SparseMatrixCSC)
    rows, cols, values = findnz(H)
    for (row, col, value) in zip(rows, cols, values)
        if isodd(value)
            chunk = ((col - 1) >>> 6) + 1
            A[row, chunk] ⊻= UInt64(1) << ((col - 1) & 63)
        end
    end
    return A
end

function _pack_mod2_rows!(A::Matrix{UInt64}, H::AbstractMatrix)
    m, n = size(H)
    for col in 1:n
        chunk = ((col - 1) >>> 6) + 1
        mask = UInt64(1) << ((col - 1) & 63)
        for row in 1:m
            if isodd(H[row, col])
                A[row, chunk] ⊻= mask
            end
        end
    end
    return A
end

function _unpack_mod2_rows(A::Matrix{UInt64}, rank::Int, n::Int)
    R = zeros(Int, rank, n)
    for col in 1:n
        chunk = ((col - 1) >>> 6) + 1
        mask = UInt64(1) << ((col - 1) & 63)
        for row in 1:rank
            R[row, col] = iszero(A[row, chunk] & mask) ? 0 : 1
        end
    end
    return R
end

function _select_independent_columns(
    G::AbstractMatrix{Int},
    r::Int,
    p::Int;
    max_iters::Int,
    rng::AbstractRNG,
    progress_io::Union{Nothing,IO},
    progress_every::Int,
    score_every::Int,
    label::AbstractString,
)
    _, n = size(G)
    perm = collect(1:n)
    best = Vector{Int}()
    best_score = Inf
    start_time = time()

    for iter in 1:max_iters
        shuffle!(rng, perm)
        cols = _independent_indices((j -> @view(G[:, j])), n, r, p; order=perm)
        if length(cols) == r && _should_score_candidate(iter, max_iters, score_every)
            score = _gram_logdet(@view(G[:, cols]))
            if score < best_score
                best_score = score
                best = copy(cols)
            end
        end
        if _should_report_progress(iter, max_iters, progress_every)
            _report_selection_progress(
                progress_io,
                _selection_progress_message(label, "selecting independent columns", iter, max_iters, start_time, best_score);
                final=iter == max_iters,
            )
        end
    end

    length(best) == r || throw(ArgumentError("could not find $(r) independent columns modulo $(p) after $(max_iters) iterations"))
    return best
end

function _select_independent_rows(
    G::AbstractMatrix{Int},
    r::Int,
    p::Int;
    max_iters::Int,
    rng::AbstractRNG,
    progress_io::Union{Nothing,IO},
    progress_every::Int,
    score_every::Int,
    label::AbstractString,
)
    k, _ = size(G)
    perm = collect(1:k)
    best = Vector{Int}()
    best_score = Inf
    start_time = time()

    for iter in 1:max_iters
        shuffle!(rng, perm)
        rows = _independent_indices((i -> @view(G[i, :])), k, r, p; order=perm)
        if length(rows) == r && _should_score_candidate(iter, max_iters, score_every)
            score = _gram_logdet(transpose(@view(G[rows, :])))
            if score < best_score
                best_score = score
                best = copy(rows)
            end
        end
        if _should_report_progress(iter, max_iters, progress_every)
            _report_selection_progress(
                progress_io,
                _selection_progress_message(label, "selecting independent rows", iter, max_iters, start_time, best_score);
                final=iter == max_iters,
            )
        end
    end

    length(best) == r || throw(ArgumentError("could not find $(r) independent rows modulo $(p) after $(max_iters) iterations"))
    return best
end

function _should_report_progress(iter::Int, max_iters::Int, progress_every::Int)
    return iter == 1 || iter == max_iters || iter % progress_every == 0
end

function _should_score_candidate(iter::Int, max_iters::Int, score_every::Int)
    return iter == 1 || iter == max_iters || iter % score_every == 0
end

function _selection_progress_message(
    label::AbstractString,
    phase::AbstractString,
    iter::Int,
    max_iters::Int,
    start_time::Float64,
    best_score::Float64,
)
    percent = round(100 * iter / max_iters; digits=1)
    elapsed = round(time() - start_time; digits=1)
    best = isfinite(best_score) ? "best_logdet=$(round(best_score; digits=3))" : "best_logdet=none"
    return "$(label) $(phase): iteration $(iter)/$(max_iters) ($(percent)%) elapsed=$(elapsed)s $(best)"
end

function _unimodular_selection_progress_message(
    label::AbstractString,
    iter::Int,
    max_iters::Int,
    start_time::Float64,
    best_logabsdet::Float64,
)
    percent = round(100 * iter / max_iters; digits=1)
    elapsed = round(time() - start_time; digits=1)
    best = isfinite(best_logabsdet) ? round(exp(best_logabsdet); digits=3) : "none"
    return "$(label) joint minor: iteration $(iter)/$(max_iters) ($(percent)%) elapsed=$(elapsed)s best_absdet=$(best)"
end

function _sparse_echelon_progress_message(
    label::AbstractString,
    step::Int,
    active::BitSet,
    rows::Vector{BitSet},
    start_time::Float64,
    last_pivot_weight::Int,
)
    elapsed = round(time() - start_time; digits=1)
    active_rows = length(active)
    active_nnz = 0
    max_weight = 0
    for row in active
        weight = length(rows[row])
        active_nnz += weight
        max_weight = max(max_weight, weight)
    end
    return "$(label) sparse echelon: pivots=$(step) active_rows=$(active_rows) active_nnz=$(active_nnz) max_row_weight=$(max_weight) last_pivot_weight=$(last_pivot_weight) elapsed=$(elapsed)s"
end

function _sparse_echelon_final_message(
    label::AbstractString,
    rank::Int,
    R::AbstractMatrix{Int},
    start_time::Float64,
)
    elapsed = round(time() - start_time; digits=1)
    return "$(label) sparse echelon: finished rank=$(rank) row_nnz=$(count(!iszero, R)) elapsed=$(elapsed)s"
end

function _report_selection_progress(progress_io::Union{Nothing,IO}, message::AbstractString; final::Bool=false)
    progress_io === nothing && return nothing

    if progress_io isa Base.TTY && !final
        print(progress_io, message, "\r")
    else
        println(progress_io, message)
    end
    flush(progress_io)

    return nothing
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

    # Keep a reduced basis with sorted pivots. Random input order can produce a
    # new pivot to the left of an existing one, so merely appending the vector
    # does not provide a valid independence test.
    for basis_vector in basis
        factor = basis_vector[pivot]
        if !iszero(factor)
            basis_vector .= mod.(basis_vector .- factor .* w, p)
        end
    end
    position = searchsortedfirst(pivots, pivot)
    insert!(basis, position, w)
    insert!(pivots, position, pivot)
    return true
end

function _gram_logdet(A)
    gram = Matrix(transpose(A) * A)
    value, sign = logabsdet(float.(gram))
    sign > 0 || return Inf
    return value
end
