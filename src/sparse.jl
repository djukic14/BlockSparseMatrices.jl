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
            _pushblocktoarrays!(block(A, blockid), rows, cols, vals)
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
            _pushblocktoarrays!(offdiagonal(A, blockid), rows, cols, vals)
        end
    end

    for color in transposeoffdiagonalcolors(A)
        for blockid in color
            _pushblocktoarrays!(transpose(offdiagonal(A, blockid)), rows, cols, vals)
        end
    end

    for color in diagonalcolors(A)
        for blockid in color
            _pushblocktoarrays!(diagonal(A, blockid), rows, cols, vals)
        end
    end

    return rows, cols, vals
end

function SparseArrays.sparse(A::AbstractBlockMatrix)
    return SparseArrays.sparse(rowcolvals(A)..., size(A)...)
end

function _pushblocktoarrays!(b, rows, cols, vals)
    for (i, rowindex) in enumerate(rowindices(b))
        for (j, colindex) in enumerate(colindices(b))
            push!(rows, rowindex)
            push!(cols, colindex)
            push!(vals, matrix(b)[i, j])
        end
    end
end
