abstract type AbstractMatrixBlock{T} <: LinearMap{T} end

function Base.eltype(block::AbstractMatrixBlock{T}) where {T}
    return eltype(matrix(block))
end

function Base.eltype(::Type{<:AbstractMatrixBlock{T}}) where {T}
    return T
end

function Base.eltype(
    ::Union{Type{<:LinearMaps.AdjointMap{T,B}},Type{<:LinearMaps.TransposeMap{T,B}}}
) where {T,B<:AbstractMatrixBlock{T}}
    return T
end

function rowindices(block::AbstractMatrixBlock)
    return block.rowindices
end

function colindices(block::AbstractMatrixBlock)
    return block.colindices
end

function matrix(block::AbstractMatrixBlock)
    return block.matrix
end

function Base.size(
    block::M
) where {
    A<:AbstractMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{<:Any,A},LinearMaps.TransposeMap{<:Any,A}},
}
    return (length(rowindices(block)), length(colindices(block)))
end

function SparseArrays.nnz(
    block::M
) where {
    A<:AbstractMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{<:Any,A},LinearMaps.TransposeMap{<:Any,A}},
}
    return nnz(matrix(block))
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, block::M, x::AbstractVector
) where {
    A<:AbstractMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{<:Any,A},LinearMaps.TransposeMap{<:Any,A}},
}
    LinearAlgebra.mul!(y, matrix(block), x, true, true)

    return y
end

function rowindices(
    block::M
) where {
    T,
    A<:AbstractMatrixBlock,
    M<:Union{LinearMaps.AdjointMap{T,A},LinearMaps.TransposeMap{T,A}},
}
    return block.lmap.colindices
end

function colindices(
    block::M
) where {
    T,
    A<:AbstractMatrixBlock,
    M<:Union{LinearMaps.AdjointMap{T,A},LinearMaps.TransposeMap{T,A}},
}
    return block.lmap.rowindices
end

function matrix(block::M) where {T,M<:LinearMaps.TransposeMap{T,<:AbstractMatrixBlock}}
    return transpose(matrix(block.lmap))
end

function matrix(block::M) where {T,M<:LinearMaps.AdjointMap{T,<:AbstractMatrixBlock}}
    return adjoint(matrix(block.lmap))
end

function Base.axes(block::AbstractMatrixBlock)
    return (Base.OneTo(maximum(rowindices(block))), Base.OneTo(maximum(colindices(block))))
end
