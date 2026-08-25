using Nemo

function _stack_gkp_generator(binary_pcm::Matrix{Int})
    row_count, column_count = size(binary_pcm)
    selected_rows = Int[]
    pivot_columns = Int[]
    basis = Dict{Int,BitVector}()

    for row_index in 1:row_count
        candidate = BitVector(!iszero(binary_pcm[row_index, column] & 1) for column in 1:column_count)

        for pivot in pivot_columns
            candidate[pivot] && (candidate .⊻= basis[pivot])
        end

        pivot = findfirst(candidate)
        pivot === nothing && continue

        # Keep the basis reduced at every pivot. Besides making subsequent
        # reductions deterministic, this ensures that a later pivot cannot
        # reintroduce an already eliminated column.
        for existing_pivot in pivot_columns
            basis[existing_pivot][pivot] && (basis[existing_pivot] .⊻= candidate)
        end
        basis[pivot] = candidate
        push!(pivot_columns, pivot)
        sort!(pivot_columns)
        push!(selected_rows, row_index)
    end

    raw_checks = binary_pcm[selected_rows, :]
    sparse_pivots = _reverse_greedy_independent_columns(raw_checks)

    if _is_unimodular_minor(raw_checks, sparse_pivots)
        independent_checks = raw_checks
        pivot_columns = sparse_pivots
    else
        # A nonsingular binary minor can have odd integer determinant larger
        # than one. In that case retaining the raw rows would generate a
        # proper sublattice, so fall back to the reduced systematic basis.
        independent_checks = zeros(Int, length(pivot_columns), column_count)
        for (row, pivot) in enumerate(pivot_columns)
            independent_checks[row, :] .= basis[pivot]
        end
    end

    gkp_columns = setdiff(1:column_count, pivot_columns)
    gkp_rows = zeros(Int, length(gkp_columns), column_count)
    for (row, column) in enumerate(gkp_columns)
        gkp_rows[row, column] = 2
    end

    return vcat(gkp_rows, independent_checks)
end

function _reverse_greedy_independent_columns(checks::Matrix{Int})
    row_count, column_count = size(checks)
    iszero(row_count) && return Int[]

    pivots = Int[]
    basis = Dict{Int,BitVector}()
    selected_columns = Int[]

    for column in column_count:-1:1
        candidate = BitVector(!iszero(checks[row, column] & 1) for row in 1:row_count)
        for pivot in pivots
            candidate[pivot] && (candidate .⊻= basis[pivot])
        end

        pivot = findfirst(candidate)
        pivot === nothing && continue
        for existing_pivot in pivots
            basis[existing_pivot][pivot] && (basis[existing_pivot] .⊻= candidate)
        end
        basis[pivot] = candidate
        push!(pivots, pivot)
        sort!(pivots)
        push!(selected_columns, column)
        length(selected_columns) == row_count && break
    end

    return selected_columns
end


function _is_unimodular_minor(checks::Matrix{Int}, columns::Vector{Int})
    row_count = size(checks, 1)
    iszero(row_count) && return true
    length(columns) == row_count || return false
    minor = matrix(ZZ, checks[:, columns])
    return abs(det(minor)) == 1
end

"""
    stack_gkp_generator(pcm)

Complete a binary parity-check matrix to a square, full-rank Construction-A
GKP lattice generator. Linearly dependent check rows are removed over
`GF(2)`, and a `2eᵢ` GKP row is added for every non-pivot column. A reverse
greedy unimodular minor is preferred so the original sparse checks are
retained. If that minor is not unimodular, a systematic basis is used to
guarantee determinant magnitude `2^(n-rank(pcm))`.

Both ordinary integer matrices and Nemo matrices over `ℤ/2ℤ` are accepted.
"""
function stack_gkp_generator(pcm::AbstractMatrix{<:Integer})
    return _stack_gkp_generator(Int.(mod.(pcm, 2)))
end

function stack_gkp_generator(pcm::Nemo.zzModMatrix)
    characteristic(base_ring(pcm)) == 2 ||
        throw(ArgumentError("stack_gkp_generator requires a matrix over a ring of characteristic 2"))
    binary_pcm = Int[Nemo.lift(pcm[row, column]) for row in 1:size(pcm, 1), column in 1:size(pcm, 2)]
    return _stack_gkp_generator(binary_pcm)
end
