"""
    struct DenseMatrixBlock{T,M,RC} <: AbstractMatrixBlock{T}

A concrete implementation of an `AbstractMatrixBlock` representing a dense matrix block.
This struct stores the actual matrix data, as well as the global row and column indices of
the block in the sparse block matrix.

# Type Parameters

  - `T`: The element type of the matrix block.
  - `M`: The type of the matrix storage.
  - `RC`: The type of the row and column index collections.

# Fields

  - `matrix`: The dense matrix stored in the block.
  - `rowindices`: The global row indices of the block in the sparse block matrix.
  - `colindices`: The global column indices of the block in the sparse block matrix.
"""
struct DenseMatrixBlock{T,M,RC} <: AbstractMatrixBlock{T}
    matrix::M
    rowindices::RC
    colindices::RC
end

"""
    DenseMatrixBlock(matrix::M, rowindices::RC, colindices::RC) where {M,RC<:Vector{Int}}

Constructs a new `DenseMatrixBlock` instance from the given matrix, row indices, and column indices.

# Arguments

  - `matrix`: The dense matrix to be stored in the block.
  - `rowindices`: The global row indices of the block in the sparse block matrix.
  - `colindices`: The global column indices of the block in the sparse block matrix.

# Type Parameters

  - `M`: The type of the matrix storage.
  - `RC`: The type of the row and column index collections, which must be a `Vector{Int}`.

# Returns

  - A new `DenseMatrixBlock` instance with the specified matrix, row indices, and column indices.

# Notes

  - The element type of the `DenseMatrixBlock` instance is inferred from the element type of the input `matrix`.
"""
function DenseMatrixBlock(
    matrix::M, rowindices::RC, colindices::RC
) where {M,RC<:Vector{Int}}
    return DenseMatrixBlock{eltype(M),M,RC}(matrix, rowindices, colindices)
end

function SparseArrays.nnz(
    block::M
) where {
    A<:DenseMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{<:Any,A},LinearMaps.TransposeMap{<:Any,A}},
}
    return length(rowindices(block)) * length(colindices(block))
end

"""
    denseblocks(blocks::Vector{M}, rowindices::V, colindices::V) where {M,V}

Constructs a vector of `DenseMatrixBlock` instances from the given blocks, row indices, and column indices.

# Arguments

  - `blocks`: A vector of matrix blocks.
  - `rowindices`: A vector of global row indices corresponding to each block.
  - `colindices`: A vector of global column indices corresponding to each block.

# Type Parameters

  - `M`: The type of the matrix blocks.
  - `V`: The type of the row and column index vectors.

# Returns

  - A vector of `DenseMatrixBlock` instances, where each instance corresponds to a block in the input `blocks` vector.

# Notes

  - The element type of the `DenseMatrixBlock` instances is inferred from the element type of the input `blocks`.
  - The `rowindices` and `colindices` vectors are assumed to have the same length as the `blocks` vector.
"""
function denseblocks(blocks::Vector{M}, rowindices::V, colindices::V) where {M,V}
    denseblockmatrices = Vector{DenseMatrixBlock{eltype(M),M,eltype(rowindices)}}(
        undef, length(blocks)
    )

    for i in eachindex(blocks)
        denseblockmatrices[i] = DenseMatrixBlock(blocks[i], rowindices[i], colindices[i])
    end
    return denseblockmatrices
end
