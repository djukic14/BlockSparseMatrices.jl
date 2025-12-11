"""
    rowcolvals(A)

Extracts the row, column, and value indices from a block-sparse matrix `A` such that a
sparse matrix can be constructed.

# Arguments

  - `A`: A block-sparse matrix.

# Returns

  - `rows`: An array of row indices.
  - `cols`: An array of column indices.
  - `vals`: An array of values.
"""
function rowcolvals(
    A::M
) where {
    Z<:BlockSparseMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    rows = Int[]
    cols = Int[]
    vals = eltype(A)[]
    for color in colors(A)
        for blockid in color
            _pushblocktoarrays!(
                rows,
                cols,
                block(A, blockid),
                rowindices(A, blockid),
                colindices(A, blockid),
                vals,
            )
        end
    end

    return rows, cols, vals
end

function rowcolvals(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    rows = Int[]
    cols = Int[]
    vals = eltype(A)[]
    for color in offdiagonalcolors(A)
        for blockid in color
            _pushblocktoarrays!(
                rows,
                cols,
                offdiagonal(A, blockid),
                rowindices(A, blockid),
                colindices(A, blockid),
                vals,
            )
        end
    end

    for color in transposeoffdiagonalcolors(A)
        for blockid in color
            _pushblocktoarrays!(
                rows,
                cols,
                transpose(offdiagonal(A, blockid)),
                colindices(A, blockid),
                rowindices(A, blockid),
                vals,
            )
        end
    end

    for color in diagonalcolors(A)
        for blockid in color
            _pushblocktoarrays!(
                rows,
                cols,
                diagonal(A, blockid),
                diagonalindices(A, blockid),
                diagonalindices(A, blockid),
                vals,
            )
        end
    end

    return rows, cols, vals
end

function rowcolvals(A::VariableBlockCompressedRowStorage)
    # Count total number of non-zero entries
    total_nnz = nnz(A)

    # Pre-allocate arrays for sparse matrix construction
    rows = Vector{Int}(undef, total_nnz)
    cols = Vector{Int}(undef, total_nnz)
    vals = Vector{eltype(A)}(undef, total_nnz)

    idx = 1
    for browidx in 1:(length(A.rowptr) - 1)
        for bidx in A.rowptr[browidx]:(A.rowptr[browidx + 1] - 1)
            block = A.blocks[bidx]
            row_start = A.rowindices[bidx]
            col_start = A.colindices[bidx]
            nrows, ncols = size(block)

            # Fill in the entries from this block
            for j in 1:ncols
                for i in 1:nrows
                    rows[idx] = row_start + i - 1
                    cols[idx] = col_start + j - 1
                    vals[idx] = block[i, j]
                    idx += 1
                end
            end
        end
    end

    return rows, cols, vals
end

#TODO: for vbcrs: write function that constructs sparse matrix directly without
# calling sparse(rowcolvals(...))
function SparseArrays.sparse(A::AbstractBlockMatrix)
    return SparseArrays.sparse(rowcolvals(A)..., size(A)...)
end

function _pushblocktoarrays!(rows, cols, b, rowindices, colindices, vals)
    for (i, rowindex) in enumerate(rowindices)
        for (j, colindex) in enumerate(colindices)
            push!(rows, rowindex)
            push!(cols, colindex)
            push!(vals, b[i, j])
        end
    end
end
