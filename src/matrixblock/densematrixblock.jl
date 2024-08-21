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
    T,
    A<:DenseMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{T,A},LinearMaps.TransposeMap{T,A}},
}
    return length(rowindices(block)) * length(colindices(block))
end
