struct SymmetricBlockMatrix{T,DM,M,D} <: AbstractBlockMatrix{T}
    diagonals::Vector{DM}
    offdiagonals::Vector{M}
    size::Tuple{Int,Int}
    forwardbuffer::Vector{T}
    adjointbuffer::Vector{T}
    buffer::Vector{T}
    diagonalsrowindexdict::D
    diagonalscolindexdict::D
    offdiagonalsrowindexdict::D
    offdiagonalscolindexdict::D
    threadsafecolors::Vector{Vector{Int}}
    ntasks::Int
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM},
    drowidcs::V,
    dcolidcs::V,
    offdiagonals::Vector{M},
    rowidcs::V,
    colidcs::V,
    size::Tuple{Int,Int};
    ntasks=1,
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

    return SymmetricBlockMatrix(diagonalblocks, offdiagonalblocks, size; ntasks=ntasks)
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM}, offdiagonals::Vector{M}, size::Tuple{Int,Int}; ntasks=1
) where {DM,M}
    return SymmetricBlockMatrix(diagonals, offdiagonals, size[1], size[2]; ntasks=ntasks)
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM}, offdiagonals::Vector{M}, rows::Int, cols::Int; ntasks=1
) where {DM,M}
    forwardbuffer, adjointbuffer, buffer = buffers(eltype(M), rows, cols)

    sort!(diagonals; lt=islessinordering)
    sort!(offdiagonals; lt=islessinordering)

    diagonalsrowindexdict = Dict{Int,Vector{Int}}()
    diagonalscolindexdict = Dict{Int,Vector{Int}}()
    for (i, block) in enumerate(diagonals)
        _appendindexdict!(diagonalsrowindexdict, block.rowindices, i)
        _appendindexdict!(diagonalscolindexdict, block.colindices, i)
    end

    offdiagonalsrowindexdict = Dict{Int,Vector{Int}}()
    offdiagonalscolindexdict = Dict{Int,Vector{Int}}()
    for (i, block) in enumerate(offdiagonals)
        _appendindexdict!(offdiagonalsrowindexdict, block.rowindices, i)
        _appendindexdict!(offdiagonalscolindexdict, block.colindices, i)
    end

    if offdiagonals != M[]
        threadsafecolors = [
            Int[] for _ in
            1:(2 * (maximum(length.(values(offdiagonalsrowindexdict))) + maximum(
                length.(values(offdiagonalscolindexdict))
            )))
        ]
    else
        threadsafecolors = Vector{Int}[]
    end
    colorperm = Vector(1:length(threadsafecolors))
    for i in eachindex(offdiagonals)
        findcolor!(
            i,
            view(threadsafecolors, colorperm),
            offdiagonals;
            threadsafecheck=issymthreadsafe(),
        )
        sortperm!(colorperm, length.(threadsafecolors))
    end

    return SymmetricBlockMatrix{eltype(M),DM,M,typeof(diagonalsrowindexdict)}(
        diagonals,
        offdiagonals,
        (rows, cols),
        forwardbuffer,
        adjointbuffer,
        buffer,
        diagonalsrowindexdict,
        diagonalscolindexdict,
        offdiagonalsrowindexdict,
        offdiagonalscolindexdict,
        threadsafecolors,
        ntasks,
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

ntasks(A::SymmetricBlockMatrix) = A.ntasks

function offdiagonal(
    A::M, i
) where {Z<:SymmetricBlockMatrix,T,M<:LinearMaps.AdjointMap{T,Z}}
    return adjoint(offdiagonal(A.lmap, i))
end

function diagonal(A::M, i) where {Z<:SymmetricBlockMatrix,T,M<:LinearMaps.AdjointMap{T,Z}}
    return adjoint(diagonal(A.lmap, i))
end

function ntasks(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    return ntasks(A.lmap)
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

function diagonalrowindices(A::SymmetricBlockMatrix, i::Integer)
    !haskey(A.diagonalsrowindexdict, i) && return Int[]
    return A.diagonalsrowindexdict[i]
end

function diagonalcolindices(A::SymmetricBlockMatrix, j::Integer)
    !haskey(A.diagonalscolindexdict, j) && return Int[]
    return A.diagonalscolindexdict[j]
end

function offdiagonalrowindices(A::SymmetricBlockMatrix, i::Integer)
    !haskey(A.offdiagonalsrowindexdict, i) && return Int[]
    return A.offdiagonalsrowindexdict[i]
end

function offdiagonalcolindices(A::SymmetricBlockMatrix, j::Integer)
    !haskey(A.offdiagonalscolindexdict, j) && return Int[]
    return A.offdiagonalscolindexdict[j]
end

threadsafecolors(A::SymmetricBlockMatrix) = A.threadsafecolors

function threadsafecolors(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return threadsafecolors(A.lmap)
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    y .= zero(eltype(y))
    for color in threadsafecolors(A)
        @tasks for blockid in color
            @set ntasks = ntasks(A)
            b = offdiagonal(A, blockid)
            LinearAlgebra.mul!(
                view(y, rowindices(b)), matrix(b), view(x, colindices(b)), true, true
            )
            LinearAlgebra.mul!(
                view(y, colindices(b)),
                transpose(matrix(b)),
                view(x, rowindices(b)),
                true,
                true,
            )
        end
    end
    @tasks for blockid in eachdiagonalindex(A)
        @set ntasks = ntasks(A)
        b = diagonal(A, blockid)
        @inbounds LinearAlgebra.mul!(view(y, rowindices(b)), b, view(x, colindices(b)))
    end
    return y
end

function Base.getindex(A::SymmetricBlockMatrix, i::Integer, j::Integer)
    (i > size(A, 1) || j > size(A, 2)) && throw(BoundsError(A, (i, j)))

    Aij = _getindex(A, i, j)
    !isnothing(Aij) && return Aij
    Aji = _getindex(A, j, i)
    !isnothing(Aji) && return Aji
    return zero(eltype(A))
end

function _getindex(A::SymmetricBlockMatrix, i::Integer, j::Integer)
    for rowblockid in diagonalrowindices(A, i)
        rowblockid ∉ diagonalcolindices(A, j) && continue
        b = diagonal(A, rowblockid)
        I = findfirst(isequal(i), rowindices(b))
        isnothing(I) && continue
        J = findfirst(isequal(j), colindices(b))
        isnothing(J) && continue
        return b.matrix[I, J]
    end

    for rowblockid in offdiagonalrowindices(A, i)
        rowblockid ∉ offdiagonalcolindices(A, j) && continue
        b = offdiagonal(A, rowblockid)
        I = findfirst(isequal(i), rowindices(b))
        isnothing(I) && continue
        J = findfirst(isequal(j), colindices(b))
        isnothing(J) && continue
        return b.matrix[I, J]
    end
    return nothing
end

function Base.setindex!(A::SymmetricBlockMatrix, v, i::Integer, j::Integer)
    (i > size(A, 1) || j > size(A, 2)) && throw(BoundsError(A, (i, j)))
    for rowblockid in diagonalrowindices(A, i)
        rowblockid ∉ diagonalcolindices(A, j) && continue
        b = diagonal(A, rowblockid)
        I = findfirst(isequal(i), rowindices(b))
        isnothing(I) && continue
        J = findfirst(isequal(j), colindices(b))
        isnothing(J) && continue
        b.matrix[I, J] = v
        b.matrix[J, I] = v
        return b.matrix[I, J]
    end

    A_ij = _setoffdiagonal!(A, v, i, j)
    !isnothing(A_ij) && return A_ij
    A_ji = _setoffdiagonal!(A, v, j, i)
    !isnothing(A_ji) && return A_ji

    return error("Value not found")
end

function _setoffdiagonal!(A::SymmetricBlockMatrix, v, i, j)
    for rowblockid in offdiagonalrowindices(A, i)
        rowblockid ∉ offdiagonalcolindices(A, j) && continue
        b = offdiagonal(A, rowblockid)
        I = findfirst(isequal(i), rowindices(b))
        isnothing(I) && continue
        J = findfirst(isequal(j), colindices(b))
        isnothing(J) && continue
        b.matrix[I, J] = v
        return b.matrix[I, J]
    end
    return nothing
end
