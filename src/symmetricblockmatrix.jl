struct SymmetricBlockMatrix{T,DM,M} <: LinearMap{T}
    diagonals::Vector{DM}
    offdiagonals::Vector{M}
    size::Tuple{Int,Int}
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

    return SymmetricBlockMatrix{eltype(M),eltype(diagonalblocks),eltype(offdiagonalblocks)}(
        diagonalblocks, offdiagonalblocks, size, Vector{eltype(M)}(undef, size[1])
    )
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM}, offdiagonals::Vector{M}, rows::Int, cols::Int
) where {DM,M}
    return SymmetricBlockMatrix{eltype(M),DM,M}(
        diagonals, offdiagonals, (rows, cols), Vector{eltype(M)}(undef, rows)
    )
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM}, offdiagonals::Vector{M}, size::Tuple{Int,Int}
) where {DM,M}
    return SymmetricBlockMatrix(diagonals, offdiagonals, size[1], size[2])
end

function Base.eltype(m::SymmetricBlockMatrix{T}) where {T}
    return eltype(typeof(m))
end

function Base.eltype(::Type{<:SymmetricBlockMatrix{T,DM,M}}) where {T,DM,M}
    return T
end

function Base.size(A::SymmetricBlockMatrix)
    return A.size
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

function buffer(A::SymmetricBlockMatrix)
    return A.buffer
end

function buffer(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return buffer(A.lmap)
end

function SparseArrays.nnz(A::SymmetricBlockMatrix)
    nonzeros = 0
    for offdiagonalid in eachoffdiagonalindex(A)
        nonzeros += nnz(offdiagonal(A, offdiagonalid))
    end
    for diagonalid in eachdiagonalindex(A)
        nonzeros += nnz(diagonal(A, diagonalid))
    end
    return nonzeros
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {
    T,
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    y .= zero(eltype(y))

    for blockid in eachoffdiagonalindex(A)
        b = offdiagonal(A, blockid)
        @views y[rowindices(b)] += matrix(b) * x[colindices(b)]
        @views y[colindices(b)] += transpose(matrix(b)) * x[rowindices(b)]
    end
    for blockid in eachdiagonalindex(A)
        b = diagonal(A, blockid)
        @inbounds LinearAlgebra.mul!(view(y, rowindices(b)), b, view(x, colindices(b)))
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector, α, β
) where {
    T,
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    temp = buffer(A)
    mul!(temp, A, x)
    y .= β .* y
    y .+= α .* temp
    return y
end
