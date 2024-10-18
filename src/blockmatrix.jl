struct BlockSparseMatrix{T,M} <: LinearMap{T}
    blocks::Vector{M}
    size::Tuple{Int,Int}
    forwardbuffer::Vector{T}
    adjointbuffer::Vector{T}
    buffer::Vector{T}
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

    buffer = Vector{eltype(M)}(undef, maximum(size))
    forwardbuffer = unsafe_wrap(typeof(buffer), pointer(buffer), size[1])
    adjointbuffer = unsafe_wrap(typeof(buffer), pointer(buffer), size[2])
    return BlockSparseMatrix{eltype(M),eltype(denseblockmatrices)}(
        denseblockmatrices, size, forwardbuffer, adjointbuffer, buffer
    )
end

function BlockSparseMatrix(blocks::Vector{M}, rows::Int, cols::Int) where {M}
    return BlockSparseMatrix{eltype(M),M}(
        blocks, (rows, cols), Vector{eltype(M)}(undef, rows)
    )
end

function BlockSparseMatrix(blocks::Vector{M}, size::Tuple{Int,Int}) where {M}
    return BlockSparseMatrix(blocks, size[1], size[2])
end

function Base.eltype(m::BlockSparseMatrix{T}) where {T}
    return eltype(typeof(m))
end

function Base.eltype(::Type{<:BlockSparseMatrix{T,M}}) where {T,M}
    return T
end

function Base.size(A::BlockSparseMatrix)
    return A.size
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

function buffer(A::BlockSparseMatrix)
    return A.forwardbuffer
end

function buffer(
    A::M
) where {
    Z<:BlockSparseMatrix,T,M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}}
}
    return A.lmap.adjointbuffer
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
    T,
    Z<:BlockSparseMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    y .= zero(eltype(y))
    for blockid in eachblockindex(A)
        b = block(A, blockid)
        @inbounds LinearAlgebra.mul!(view(y, rowindices(b)), b, view(x, colindices(b)))
    end
    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector, α, β
) where {
    T,
    Z<:BlockSparseMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    temp = buffer(A)
    mul!(temp, A, x)
    y .= β .* y
    y .+= α .* temp
    return y
end

function Base.convert(::SparseMatrixCSC, M::BlockSparseMatrix)
    rows, cols = size(M)
    nnz = SparseArrays.nnz(M)
    rowptr = Vector{Int}(undef, rows + 1)
    colidx = Vector{Int}(undef, nnz)
    values = Vector{eltype(M)}(undef, nnz)

    rowptr[1] = 1
    row = 1
    for blockid in eachblockindex(M)
        block = block(M, blockid)
        blockrows, blockcols = size(block)
        for blockrow in 1:blockrows
            for blockcol in 1:blockcols
                col = blockcol + blockcolindices(block) - 1
                colidx[row] = col
                values[row] = block[blockrow, blockcol]
                row += 1
            end
        end
        rowptr[blockrowindices(block) + 1] = row
    end

    return SparseMatrixCSC(rows, cols, rowptr, colidx, values)
end
