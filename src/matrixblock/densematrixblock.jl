"""
    DenseMatrixBlock{T,M,RC} <: AbstractMatrixBlock{T}

Dense block in a sparse blockmatrix.

# Fields
- `matrix::M`: Matrix.
- `rowindices::RC`: Global row indices.
- `colindices::RC`: Global column indices.
"""
struct DenseMatrixBlock{T,M,RC} <: AbstractMatrixBlock{T}
    matrix::M
    rowindices::RC
    colindices::RC
end

function DenseMatrixBlock(
    matrix::M, rowindices::RC, colindices::RC
) where {M,RC<:AbstractVector{Int}}
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
