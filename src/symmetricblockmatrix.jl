"""
    struct SymmetricBlockMatrix{T,DM,M,D,S} <: AbstractBlockMatrix{T}

A concrete implementation of a symmetric block matrix, which is a block matrix where the
off-diagonal blocks are shared between the upper and lower triangular parts.
The diagonal blocks are symmetric as well.

# Type Parameters

  - `T`: The element type of the matrix.
  - `DM`: The type of the diagonal matrix blocks.
  - `M`: The type of the off-diagonal matrix blocks.
  - `D`: The type of the row and column index dictionaries.
  - `S`: The type of the scheduler.

# Fields

  - `diagonals`: A vector of diagonal matrix blocks.
  - `offdiagonals`: A vector of off-diagonal matrix blocks.
  - `size`: A tuple representing the size of the symmetric block matrix.
  - `forwardbuffer`: A buffer used for forward matrix-vector product computations.
  - `adjointbuffer`: A buffer used for adjoint matrix-vector product computations.
  - `buffer`: The underlying buffer that is reused for both forward and adjoint products.
  - `diagonalsrowindexdict`: A dictionary that maps row indices to diagonal block indices.
  - `diagonalscolindexdict`: A dictionary that maps column indices to diagonal block indices.
  - `offdiagonalsrowindexdict`: A dictionary that maps row indices to off-diagonal block
    indices.
  - `offdiagonalscolindexdict`: A dictionary that maps column indices to off-diagonal block
    indices.
  - `diagonalcolors`: A vector of colors for the diagonal blocks, where each color is a vector
    of block indices that can be processed in parallel without race conditions.
  - `offdiagonalcolors`: A vector of colors for the off-diagonal blocks, where each color is a
    vector of block indices that can be processed in parallel without race conditions.
  - `transposeoffdiagonalcolors`: A vector of colors for the transposed off-diagonal blocks,
    where each color is a vector of block indices that can be processed in parallel without
    race conditions.
  - `scheduler`: A scheduler that manages the parallel computation of matrix-vector products.
"""
struct SymmetricBlockMatrix{T,DM,M,D,S} <: AbstractBlockMatrix{T}
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
    diagonalcolors::Vector{Vector{Int}}
    offdiagonalcolors::Vector{Vector{Int}}
    transposeoffdiagonalcolors::Vector{Vector{Int}}
    scheduler::S
end

"""
    SymmetricBlockMatrix(
        diagonals::Vector{DM},
        diagonalindices::V,
        offdiagonals::Vector{M},
        rowindices::V,
        columnindices::V,
        size::Tuple{Int,Int};
        scheduler=DynamicScheduler(),
    ) where {DM,M,V}

Constructs a new `SymmetricBlockMatrix` instance from the given diagonal and off-diagonal blocks, indices, and size.

# Arguments

  - `diagonals`: A vector of symmetric dense matrices.
  - `diagonalindices`: A vector of indices corresponding to the diagonal blocks.
  - `offdiagonals`: A vector of dense matrices.
  - `rowindices`: A vector of row indices corresponding to the off-diagonal blocks.
  - `columnindices`: A vector of column indices corresponding to the off-diagonal blocks.
  - `size`: A tuple representing the size of the symmetric block matrix.
  - `scheduler`: The scheduler used to manage parallel computation. Defaults to `DynamicScheduler()`.

# Returns

  - A new `SymmetricBlockMatrix` instance constructed from the given blocks, indices, and size.
"""
function SymmetricBlockMatrix(
    diagonals::Vector{DM},
    diagonalindices::V,
    offdiagonals::Vector{M},
    rowindices::V,
    columnindices::V,
    size::Tuple{Int,Int};
    scheduler=DynamicScheduler(),
) where {DM,M,V}
    return SymmetricBlockMatrix(
        denseblocks(diagonals, diagonalindices, diagonalindices),
        denseblocks(offdiagonals, rowindices, columnindices),
        size;
        scheduler=scheduler,
    )
end

"""
    SymmetricBlockMatrix(
        diagonals::Vector{DM},
        offdiagonals::Vector{M},
        size::Tuple{Int,Int};
        scheduler=DynamicScheduler(),
    ) where {DM<:AbstractMatrixBlock,M<:AbstractMatrixBlock}

Constructs a new `SymmetricBlockMatrix` instance from the given diagonal and off-diagonal blocks, and size.

# Arguments

  - `diagonals`: A vector of `AbstractMatrixBlock` instances.
  - `offdiagonals`: A vector of `AbstractMatrixBlock` instances.
  - `size`: A tuple representing the size of the symmetric block matrix.
  - `scheduler`: The scheduler used to manage parallel computation. Defaults to `DynamicScheduler()`.

# Returns

  - A new `SymmetricBlockMatrix` instance constructed from the given blocks and size.
"""
function SymmetricBlockMatrix(
    diagonals::Vector{DM},
    offdiagonals::Vector{M},
    size::Tuple{Int,Int};
    scheduler=DynamicScheduler(),
) where {DM<:AbstractMatrixBlock,M<:AbstractMatrixBlock}
    return SymmetricBlockMatrix(
        diagonals, offdiagonals, size[1], size[2]; scheduler=scheduler
    )
end

function SymmetricBlockMatrix(
    diagonals::Vector{DM},
    offdiagonals::Vector{M},
    rows::Int,
    cols::Int;
    scheduler=SerialScheduler(),
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

    diagonalcolors = color(conflictgraph(diagonals); algorithm=coloringalgorithm).colors

    offdiagonalcolors =
        color(conflictgraph(offdiagonals); algorithm=coloringalgorithm).colors
    transposeoffdiagonalcolors =
        color(conflictgraph(offdiagonals; transpose=true); algorithm=coloringalgorithm).colors

    return SymmetricBlockMatrix{
        eltype(M),DM,M,typeof(diagonalsrowindexdict),typeof(scheduler)
    }(
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
        diagonalcolors,
        offdiagonalcolors,
        transposeoffdiagonalcolors,
        scheduler,
    )
end

"""
    eachoffdiagonalindex(A::SymmetricBlockMatrix)

Returns an iterator over the indices of the off-diagonal blocks in the given
`SymmetricBlockMatrix` instance.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.

# Returns

  - An iterator that yields the indices of the off-diagonal blocks in the `SymmetricBlockMatrix`.
"""
function eachoffdiagonalindex(A::SymmetricBlockMatrix)
    return eachindex(A.offdiagonals)
end

"""
    eachdiagonalindex(A::SymmetricBlockMatrix)

Returns an iterator over the indices of the diagonal blocks in the given `SymmetricBlockMatrix` instance.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.

# Returns

  - An iterator that yields the indices of the diagonal blocks in the `SymmetricBlockMatrix`.
"""
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

"""
    offdiagonal(A::SymmetricBlockMatrix, i)

Returns the `i`-th off-diagonal block of the given `SymmetricBlockMatrix` instance.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.
  - `i`: The index of the off-diagonal block to retrieve.

# Returns

  - The `i`-th off-diagonal block of the `SymmetricBlockMatrix`.
"""
function offdiagonal(A::SymmetricBlockMatrix, i)
    return A.offdiagonals[i]
end

"""
    diagonal(A::SymmetricBlockMatrix, i)

Returns the `i`-th diagonal block of the given `SymmetricBlockMatrix` instance.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.
  - `i`: The index of the diagonal block to retrieve.

# Returns

  - The `i`-th diagonal block of the `SymmetricBlockMatrix`.
"""
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

"""
    diagonalcolors(A::SymmetricBlockMatrix)

Returns the colors used for the diagonal blocks of the given `SymmetricBlockMatrix` instance.
These colors are used to coordinate parallel computation and avoid race conditions.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.

# Returns

  - A vector of colors, where each color is a vector of diagonal block indices that can be
    processed in parallel without race conditions.
"""
function diagonalcolors(A::SymmetricBlockMatrix)
    return A.diagonalcolors
end

function diagonalcolors(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return diagonalcolors(A.lmap)
end

"""
    offdiagonalcolors(A::SymmetricBlockMatrix)

Returns the colors used for the off-diagonal blocks of the given `SymmetricBlockMatrix`
instance. These colors are used to coordinate parallel computation and avoid race conditions.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.

# Returns

  - A vector of colors, where each color is a vector of off-diagonal block indices that can be
    processed in parallel without race conditions.
"""
function offdiagonalcolors(A::SymmetricBlockMatrix)
    return A.offdiagonalcolors
end

"""
    transposeoffdiagonalcolors(A::SymmetricBlockMatrix)

Returns the colors used for the transposed off-diagonal blocks of the given
`SymmetricBlockMatrix` instance. These colors are used to coordinate parallel computation
and avoid race conditions when computing the transpose of the matrix.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.

# Returns

  - A vector of colors, where each color is a vector of transposed off-diagonal block indices
    that can be processed in parallel without race conditions.
"""
function transposeoffdiagonalcolors(A::SymmetricBlockMatrix)
    return A.transposeoffdiagonalcolors
end

function transposeoffdiagonalcolors(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return offdiagonalcolors(A.lmap)
end

function offdiagonalcolors(
    A::M
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return transposeoffdiagonalcolors(A.lmap)
end

function SparseArrays.nnz(
    A::M
) where {
    M<:Union{
        SymmetricBlockMatrix,
        LinearMaps.AdjointMap{<:Any,<:SymmetricBlockMatrix},
        LinearMaps.TransposeMap{<:Any,<:SymmetricBlockMatrix},
    },
}
    nonzeros = 0
    for offdiagonalid in eachoffdiagonalindex(A)
        nonzeros += 2 * nnz(offdiagonal(A, offdiagonalid))
    end
    for diagonalid in eachdiagonalindex(A)
        nonzeros += nnz(diagonal(A, diagonalid))
    end
    return nonzeros
end

"""
    diagonalrowindices(A::SymmetricBlockMatrix, i::Integer)

Returns the indices of the diagonal blocks in the given `SymmetricBlockMatrix` instance that
contain the row index `i`.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.
  - `i`: The row index to search for.

# Returns

  - A vector of indices of the diagonal blocks that contain the row index `i`. If no such
    blocks exist, an empty vector is returned.
"""
function diagonalrowindices(A::SymmetricBlockMatrix, i::Integer)
    !haskey(A.diagonalsrowindexdict, i) && return Int[]
    return A.diagonalsrowindexdict[i]
end

"""
    diagonalcolindices(A::SymmetricBlockMatrix, j::Integer)

Returns the indices of the diagonal blocks in the given `SymmetricBlockMatrix` instance that
contain the column index `j`.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.
  - `j`: The column index to search for.

# Returns

  - A vector of indices of the diagonal blocks that contain the column index `j`. If no such
    blocks exist, an empty vector is returned.
"""
function diagonalcolindices(A::SymmetricBlockMatrix, j::Integer)
    !haskey(A.diagonalscolindexdict, j) && return Int[]
    return A.diagonalscolindexdict[j]
end

"""
    offdiagonalrowindices(A::SymmetricBlockMatrix, i::Integer)

Returns the indices of the off-diagonal blocks in the given `SymmetricBlockMatrix` instance
that contain the row index `i`.

# Arguments

  - `A`: The `SymmetricBlockMatrix` instance to query.
  - `i`: The row index to search for.

# Returns

  - A vector of indices of the off-diagonal blocks that contain the row index `i`. If no such
    blocks exist, an empty vector is returned.
"""
function offdiagonalrowindices(A::SymmetricBlockMatrix, i::Integer)
    !haskey(A.offdiagonalsrowindexdict, i) && return Int[]
    return A.offdiagonalsrowindexdict[i]
end

"""
    offdiagonalcolindices(A::SymmetricBlockMatrix, j::Integer)

Returns the indices of the off-diagonal blocks in the given `SymmetricBlockMatrix` instance
that contain the column index `j`.

# Arguments
- `A`: The `SymmetricBlockMatrix` instance to query.
- `j`: The column index to search for.

# Returns
- A vector of indices of the off-diagonal blocks that contain the column index `j`. If no
such blocks exist, an empty vector is returned.
"""

function offdiagonalcolindices(A::SymmetricBlockMatrix, j::Integer)
    !haskey(A.offdiagonalscolindexdict, j) && return Int[]
    return A.offdiagonalscolindexdict[j]
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    y .= zero(eltype(y))
    for color in offdiagonalcolors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            b = offdiagonal(A, blockid)
            LinearAlgebra.mul!(
                view(y, rowindices(b)), matrix(b), view(x, colindices(b)), true, true
            )
        end
    end

    for color in transposeoffdiagonalcolors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            b = offdiagonal(A, blockid)

            LinearAlgebra.mul!(
                view(y, colindices(b)),
                transpose(matrix(b)),
                view(x, rowindices(b)),
                true,
                true,
            )
        end
    end

    for color in diagonalcolors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            b = diagonal(A, blockid)
            LinearAlgebra.mul!(
                view(y, rowindices(b)), matrix(b), view(x, colindices(b)), true, true
            )
        end
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
