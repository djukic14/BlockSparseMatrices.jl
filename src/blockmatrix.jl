struct BlockSparseMatrix{T,M,D} <: AbstractBlockMatrix{T}
    blocks::Vector{M}
    size::Tuple{Int,Int}
    forwardbuffer::Vector{T}
    adjointbuffer::Vector{T}
    buffer::Vector{T}
    rowindexdict::D
    colindexdict::D
    threadsafecolors::Vector{Vector{Int}}
    ntasks::Int
end

function BlockSparseMatrix(
    blocks::Vector{M}, rowindices::V, colindices::V, size::Tuple{Int,Int}; ntasks=1
) where {M,V}
    denseblockmatrices = Vector{DenseMatrixBlock{eltype(M),M,eltype(rowindices)}}(
        undef, length(blocks)
    )

    for i in eachindex(blocks)
        denseblockmatrices[i] = DenseMatrixBlock(blocks[i], rowindices[i], colindices[i])
    end

    return BlockSparseMatrix(denseblockmatrices, size; ntasks=ntasks)
end

function BlockSparseMatrix(blocks::Vector{M}, size::Tuple{Int,Int}; ntasks=1) where {M}
    return BlockSparseMatrix(blocks, size[1], size[2]; ntasks=ntasks)
end

function BlockSparseMatrix(blocks::Vector{M}, rows::Int, cols::Int; ntasks=1) where {M}
    forwardbuffer, adjointbuffer, buffer = buffers(eltype(M), rows, cols)

    sort!(blocks; lt=islessinordering)

    rowindexdict = Dict{Int,Vector{Int}}()
    colindexdict = Dict{Int,Vector{Int}}()

    for (i, block) in enumerate(blocks)
        _appendindexdict!(rowindexdict, block.rowindices, i)
        _appendindexdict!(colindexdict, block.colindices, i)
    end

    #TODO: Pessimistic choice -> check performance
    if blocks != M[]
        threadsafecolors = [
            Int[] for _ in
            1:(maximum(length.(values(rowindexdict))) + maximum(
                length.(values(colindexdict))
            ))
        ]
    else
        threadsafecolors = Vector{Int}[]
    end
    colorperm = Vector(1:length(threadsafecolors))
    for i in eachindex(blocks)
        findcolor!(i, view(threadsafecolors, colorperm), blocks)
        sortperm!(colorperm, length.(threadsafecolors))
    end

    return BlockSparseMatrix{eltype(M),M,typeof(rowindexdict)}(
        blocks,
        (rows, cols),
        forwardbuffer,
        adjointbuffer,
        buffer,
        rowindexdict,
        colindexdict,
        threadsafecolors,
        ntasks,
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

ntasks(A::BlockSparseMatrix) = A.ntasks

function ntasks(
    A::M
) where {
    Z<:BlockSparseMatrix,T,M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}}
}
    return ntasks(A.lmap)
end

threadsafecolors(A::BlockSparseMatrix) = A.threadsafecolors

function threadsafecolors(
    A::M
) where {
    Z<:BlockSparseMatrix,T,M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}}
}
    return threadsafecolors(A.lmap)
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
    for color in threadsafecolors(A)
        @tasks for blockid in color
            @set ntasks = ntasks(A)
            b = block(A, blockid)
            @inbounds LinearAlgebra.mul!(view(y, rowindices(b)), b, view(x, colindices(b)))
        end
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
