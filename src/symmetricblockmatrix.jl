"""
    struct SymmetricBlockMatrix{T,D,P,M,S} <: AbstractBlockMatrix{T}

A concrete implementation of a symmetric block matrix, which is a block matrix where the
off-diagonal blocks are shared between the upper and lower triangular parts.
The diagonal blocks are symmetric as well.

# Type Parameters

  - `T`: The element type of the matrix.
  - `D`: The type of the diagonal matrix blocks.
  - `P`: The integer type used for indexing.
  - `M`: The type of the off-diagonal matrix blocks.
  - `S`: The type of the scheduler.

# Fields

  - `diagonals`: A vector of diagonal matrix blocks.
  - `diagonalindices`: A vector where each element is a vector of indices for the corresponding diagonal block.
  - `offdiagonals`: A vector of off-diagonal matrix blocks.
  - `rowindices`: A vector where each element is a vector of row indices for the corresponding off-diagonal block.
  - `colindices`: A vector where each element is a vector of column indices for the corresponding off-diagonal block.
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
struct SymmetricBlockMatrix{T,D,P,M,S} <: AbstractBlockMatrix{T}
    diagonals::Vector{D}
    diagonalindices::Vector{Vector{P}}
    offdiagonals::Vector{M}
    rowindices::Vector{Vector{P}}
    colindices::Vector{Vector{P}}
    size::Tuple{Int,Int}
    diagonalcolors::Vector{Vector{Int}}
    offdiagonalcolors::Vector{Vector{Int}}
    transposeoffdiagonalcolors::Vector{Vector{Int}}
    scheduler::S
end

"""
    SymmetricBlockMatrix(
        diagonals,
        diagonalindices,
        offdiagonals,
        rowindices,
        colindices,
        size::Tuple{Int,Int};
        scheduler=DynamicScheduler(),
    )

Constructs a new `SymmetricBlockMatrix` instance from the given diagonal and off-diagonal blocks with their indices.

# Arguments

  - `diagonals`: A vector of diagonal matrix blocks.
  - `diagonalindices`: A vector where each element is a vector of indices for the corresponding diagonal block.
  - `offdiagonals`: A vector of off-diagonal matrix blocks.
  - `rowindices`: A vector where each element is a vector of row indices for the corresponding off-diagonal block.
  - `colindices`: A vector where each element is a vector of column indices for the corresponding off-diagonal block.
  - `size`: A tuple representing the size of the symmetric block matrix.
  - `scheduler`: The scheduler used to manage parallel computation. Defaults to `DynamicScheduler()`.

# Returns

  - A new `SymmetricBlockMatrix` instance constructed from the given blocks and indices.
"""
function SymmetricBlockMatrix(
    diagonals,
    diagonalindices,
    offdiagonals,
    rowindices,
    colindices,
    size::Tuple{Int,Int};
    scheduler=DynamicScheduler(),
)
    return SymmetricBlockMatrix(
        diagonals,
        diagonalindices,
        offdiagonals,
        rowindices,
        colindices,
        size[1],
        size[2];
        scheduler=scheduler,
    )
end

function SymmetricBlockMatrix(
    diagonals::Vector{D},
    diagonalindices,
    offdiagonals::Vector{M},
    rowindices,
    colindices,
    rows::Int,
    cols::Int;
    scheduler=SerialScheduler(),
) where {D,M}
    diagonalcolors =
        color(conflictgraph(ColorInfo(diagonalindices)); algorithm=coloringalgorithm).colors

    offdiagonalcolors =
        color(conflictgraph(ColorInfo(rowindices)); algorithm=coloringalgorithm).colors
    transposeoffdiagonalcolors =
        color(conflictgraph(ColorInfo(colindices)); algorithm=coloringalgorithm).colors

    return SymmetricBlockMatrix{
        eltype(eltype(D)),D,eltype(eltype(rowindices)),M,typeof(scheduler)
    }(
        diagonals,
        diagonalindices,
        offdiagonals,
        rowindices,
        colindices,
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

function diagonalindices(A::SymmetricBlockMatrix, blockid)
    return A.diagonalindices[blockid]
end

function diagonalindices(
    A::M, blockid
) where {
    Z<:SymmetricBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return diagonalindices(A.lmap, blockid)
end

function rowindices(A::Union{BlockSparseMatrix,SymmetricBlockMatrix}, blockid)
    return A.rowindices[blockid]
end

function rowindices(
    A::M, blockid
) where {
    Z<:Union{BlockSparseMatrix,SymmetricBlockMatrix},
    M<:Union{LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    return A.lmap.colindices[blockid]
end

function colindices(A::Union{BlockSparseMatrix,SymmetricBlockMatrix}, blockid)
    return A.colindices[blockid]
end

function colindices(
    A::M, blockid
) where {
    Z<:Union{BlockSparseMatrix,SymmetricBlockMatrix},
    M<:Union{LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    return A.lmap.rowindices[blockid]
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
        nonzeros += 2 * _nnz(offdiagonal(A, offdiagonalid))
    end
    for diagonalid in eachdiagonalindex(A)
        nonzeros += _nnz(diagonal(A, diagonalid))
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
            LinearAlgebra.mul!(
                view(y, rowindices(A, blockid)),
                offdiagonal(A, blockid),
                view(x, colindices(A, blockid)),
                α,
                true,
            )
        end
    end

    for color in transposeoffdiagonalcolors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)
            LinearAlgebra.mul!(
                view(y, colindices(A, blockid)),
                transpose(offdiagonal(A, blockid)),
                view(x, rowindices(A, blockid)),
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
                view(y, diagonalindices(A, blockid)),
                b,
                view(x, diagonalindices(A, blockid)),
                α,
                true,
            )
        end
    end

    return y
end
