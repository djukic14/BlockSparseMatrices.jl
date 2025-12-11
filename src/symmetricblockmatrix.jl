"""
    struct SymmetricBlockMatrix{T,D,M,D,S} <: AbstractBlockMatrix{T}

A concrete implementation of a symmetric block matrix, which is a block matrix where the
off-diagonal blocks are shared between the upper and lower triangular parts.
The diagonal blocks are symmetric as well.

# Type Parameters

  - `T`: The element type of the matrix.
  - `D`: The type of the diagonal matrix blocks.
  - `M`: The type of the off-diagonal matrix blocks.
  - `S`: The type of the scheduler.

# Fields

  - `diagonals`: A vector of diagonal matrix blocks.
  - `offdiagonals`: A vector of off-diagonal matrix blocks.
  - `size`: A tuple representing the size of the symmetric block matrix.
  - `diagonalcolors`: A vector of colors for the diagonal blocks, where each color is a vector
    of block indices that can be processed in parallel without race conditions.
  - `offdiagonalcolors`: A vector of colors for the off-diagonal blocks, where each color is a
    vector of block indices that can be processed in parallel without race conditions.
  - `transposeoffdiagonalcolors`: A vector of colors for the transposed off-diagonal blocks,
    where each color is a vector of block indices that can be processed in parallel without
    race conditions.
  - `scheduler`: A scheduler that manages the parallel computation of matrix-vector products.
"""
struct SymmetricBlockMatrix{T,D,M,S} <: AbstractBlockMatrix{T}
    diagonals::Vector{D}
    offdiagonals::Vector{M}
    size::Tuple{Int,Int}
    diagonalcolors::Vector{Vector{Int}}
    offdiagonalcolors::Vector{Vector{Int}}
    transposeoffdiagonalcolors::Vector{Vector{Int}}
    scheduler::S
end

"""
    SymmetricBlockMatrix(
        diagonals::Vector{D},
        diagonalindices::V,
        offdiagonals::Vector{M},
        rowindices::V,
        columnindices::V,
        size::Tuple{Int,Int};
        scheduler=DynamicScheduler(),
    ) where {D,M,V}

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
    diagonals::Vector{D},
    diagonalindices::V,
    offdiagonals::Vector{M},
    rowindices::V,
    columnindices::V,
    size::Tuple{Int,Int};
    scheduler=DynamicScheduler(),
) where {D,M,V}
    return SymmetricBlockMatrix(
        denseblocks(diagonals, diagonalindices, diagonalindices),
        denseblocks(offdiagonals, rowindices, columnindices),
        size;
        scheduler=scheduler,
    )
end

"""
    SymmetricBlockMatrix(
        diagonals::Vector{D},
        offdiagonals::Vector{M},
        size::Tuple{Int,Int};
        scheduler=DynamicScheduler(),
    ) where {D<:AbstractMatrixBlock,M<:AbstractMatrixBlock}

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
    diagonals::Vector{D},
    offdiagonals::Vector{M},
    size::Tuple{Int,Int};
    scheduler=DynamicScheduler(),
) where {D<:AbstractMatrixBlock,M<:AbstractMatrixBlock}
    return SymmetricBlockMatrix(
        diagonals, offdiagonals, size[1], size[2]; scheduler=scheduler
    )
end

function SymmetricBlockMatrix(
    diagonals::Vector{D},
    offdiagonals::Vector{M},
    rows::Int,
    cols::Int;
    scheduler=SerialScheduler(),
) where {D,M}
    sort!(diagonals; lt=islessinordering)
    sort!(offdiagonals; lt=islessinordering)

    diagonalcolors = color(conflictgraph(diagonals); algorithm=coloringalgorithm).colors

    offdiagonalcolors =
        color(conflictgraph(offdiagonals); algorithm=coloringalgorithm).colors
    transposeoffdiagonalcolors =
        color(conflictgraph(offdiagonals; transpose=true); algorithm=coloringalgorithm).colors

    return SymmetricBlockMatrix{eltype(M),D,M,typeof(scheduler)}(
        diagonals,
        offdiagonals,
        (rows, cols),
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

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector, α::Number, β::Number
) where {
    Z<:SymmetricBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    y .*= β

    for color in offdiagonalcolors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            b = offdiagonal(A, blockid)
            LinearAlgebra.mul!(
                view(y, rowindices(b)), matrix(b), view(x, colindices(b)), α, true
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
                α,
                true,
            )
        end
    end

    for color in diagonalcolors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            b = diagonal(A, blockid)
            LinearAlgebra.mul!(
                view(y, rowindices(b)), matrix(b), view(x, colindices(b)), α, true
            )
        end
    end

    return y
end
