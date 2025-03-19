struct BlockSparseMatrix{T,M,D} <: AbstractBlockMatrix{T}
    blocks::Vector{M}
    size::Tuple{Int,Int}
    forwardbuffer::Vector{T}
    adjointbuffer::Vector{T}
    buffer::Vector{T}
    rowindexdict::D
    colindexdict::D
end

function BlockSparseMatrix(
    blocks::Vector{M}, rowindices::V, colindices::V, size::Tuple{Int,Int}
) where {M,V}
    denseblockmatrices = Vector{DenseMatrixBlock{eltype(M),M,eltype(rowindices)}}(
        undef, length(blocks)
    )

    for i in eachindex(blocks)
        denseblockmatrices[i] = DenseMatrixBlock(blocks[i], rowindices[i], colindices[i])
    end

    return BlockSparseMatrix(denseblockmatrices, size)
end

function BlockSparseMatrix(blocks::Vector{M}, size::Tuple{Int,Int}) where {M}
    return BlockSparseMatrix(blocks, size[1], size[2])
end

function BlockSparseMatrix(blocks::Vector{M}, rows::Int, cols::Int) where {M}
    forwardbuffer, adjointbuffer, buffer = buffers(eltype(M), rows, cols)

    sort!(blocks; lt=islessinordering)

    rowindexdict = Dict{Int,Vector{Int}}()
    colindexdict = Dict{Int,Vector{Int}}()

    for (i, block) in enumerate(blocks)
        _appendindexdict!(rowindexdict, block.rowindices, i)
        _appendindexdict!(colindexdict, block.colindices, i)
    end

    return BlockSparseMatrix{eltype(M),M,typeof(rowindexdict)}(
        blocks,
        (rows, cols),
        forwardbuffer,
        adjointbuffer,
        buffer,
        rowindexdict,
        colindexdict,
    )
end

function eachblockindex(A::BlockSparseMatrix)
    return eachindex(A.blocks)
end

function eachblockindex(
    A::M
) where {
    Z<:BlockSparseMatrix,T,M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}}
}
    return eachblockindex(A.lmap)
end

function block(A::BlockSparseMatrix, i)
    return A.blocks[i]
end

function block(A::M, i) where {Z<:BlockSparseMatrix,T,M<:LinearMaps.AdjointMap{T,Z}}
    return adjoint(block(A.lmap, i))
end

function block(A::M, i) where {Z<:BlockSparseMatrix,T,M<:LinearMaps.TransposeMap{T,Z}}
    return transpose(block(A.lmap, i))
end

function SparseArrays.nnz(A::BlockSparseMatrix)
    nonzeros = 0
    for blockid in eachblockindex(A)
        nonzeros += nnz(block(A, blockid))
    end

    return nonzeros
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {
    Z<:BlockSparseMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    y .= zero(eltype(y))
    for blockid in eachblockindex(A)
        b = block(A, blockid)
        @inbounds LinearAlgebra.mul!(view(y, rowindices(b)), b, view(x, colindices(b)))
    end
    return y
end

function _appendindexdict!(dict, indices, blockid)
    for index in indices
        !haskey(dict, index) && (dict[index] = [])
        push!(dict[index], blockid)
    end
    return dict
end
