struct SymmetricBlockMatrix{T,DM,M} <: AbstractBlockMatrix{T}
    diagonals::Vector{DM}
    offdiagonals::Vector{M}
    size::Tuple{Int,Int}
    forwardbuffer::Vector{T}
    adjointbuffer::Vector{T}
    buffer::Vector{T}
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM},
    drowidcs::V,
    dcolidcs::V,
    offdiagonals::Vector{M},
    rowidcs::V,
    colidcs::V,
    size::Tuple{Int,Int},
) where {DM,M,V}
    offdiagonalblocks = Vector{DenseMatrixBlock{eltype(M),M,eltype(rowidcs)}}(
        undef, length(offdiagonals)
    )
    diagonalblocks = Vector{DenseMatrixBlock{eltype(M),M,eltype(rowidcs)}}(
        undef, length(diagonals)
    )

    for i in eachindex(offdiagonals)
        offdiagonalblocks[i] = DenseMatrixBlock(offdiagonals[i], rowidcs[i], colidcs[i])
    end

    for i in eachindex(diagonals)
        diagonalblocks[i] = DenseMatrixBlock(diagonals[i], drowidcs[i], dcolidcs[i])
    end

    return SymmetricBlockMatrix(diagonalblocks, offdiagonalblocks, size)
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM}, offdiagonals::Vector{M}, size::Tuple{Int,Int}
) where {DM,M}
    return SymmetricBlockMatrix(diagonals, offdiagonals, size[1], size[2])
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM}, offdiagonals::Vector{M}, rows::Int, cols::Int
) where {DM,M}
    forwardbuffer, adjointbuffer, buffer = buffers(eltype(M), rows, cols)
    return SymmetricBlockMatrix{eltype(M),DM,M}(
        diagonals, offdiagonals, (rows, cols), forwardbuffer, adjointbuffer, buffer
    )
end

function eachoffdiagonalindex(A::SymmetricBlockMatrix)
    return eachindex(A.offdiagonals)
end

function eachdiagonalindex(A::SymmetricBlockMatrix)
    return eachindex(A.diagonals)
end

function eachoffdiagonalindex(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return eachoffdiagonalindex(A.lmap)
end

function eachdiagonalindex(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return eachdiagonalindex(A.lmap)
end

function offdiagonal(A::SymmetricBlockMatrix, i)
    return A.offdiagonals[i]
end

function diagonal(A::SymmetricBlockMatrix, i)
    return A.diagonals[i]
end

function offdiagonal(
    A::M, i
) where {Z<:SymmetricBlockMatrix,T,M<:LinearMaps.AdjointMap{T,Z}}
    return adjoint(offdiagonal(A.lmap, i))
end

function diagonal(A::M, i) where {Z<:SymmetricBlockMatrix,T,M<:LinearMaps.AdjointMap{T,Z}}
    return adjoint(diagonal(A.lmap, i))
end

function offdiagonal(
    A::M, i
) where {Z<:SymmetricBlockMatrix,T,M<:LinearMaps.TransposeMap{T,Z}}
    return transpose(offdiagonal(A.lmap, i))
end

function diagonal(A::M, i) where {Z<:SymmetricBlockMatrix,T,M<:LinearMaps.TransposeMap{T,Z}}
    return transpose(diagonal(A.lmap, i))
end

function SparseArrays.nnz(A::SymmetricBlockMatrix)
    nonzeros = 0
    for offdiagonalid in eachoffdiagonalindex(A)
        nonzeros += 2 * nnz(offdiagonal(A, offdiagonalid))
    end
    for diagonalid in eachdiagonalindex(A)
        nonzeros += nnz(diagonal(A, diagonalid))
    end
    return nonzeros
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    y .= zero(eltype(y))
    for blockid in eachoffdiagonalindex(A)
        b = offdiagonal(A, blockid)
        LinearAlgebra.mul!(
            view(y, rowindices(b)), matrix(b), view(x, colindices(b)), true, true
        )
        LinearAlgebra.mul!(
            view(y, colindices(b)), transpose(matrix(b)), view(x, rowindices(b)), true, true
        )
    end
    for blockid in eachdiagonalindex(A)
        b = diagonal(A, blockid)
        @inbounds LinearAlgebra.mul!(view(y, rowindices(b)), b, view(x, colindices(b)))
    end
    return y
end
