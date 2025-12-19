"""
    struct BlockSparseMatrix{T,M,P,S} <: AbstractBlockMatrix{T}

A concrete implementation of a block sparse matrix, which is a sparse matrix composed of
smaller dense matrix blocks.

# Type Parameters

  - `T`: The element type of the matrix.
  - `M`: The type of the matrix blocks.
  - `P`: The integer type used for indexing.
  - `S`: The type of the scheduler.

# Fields

  - `blocks`: A vector of matrix blocks that comprise the block sparse matrix.
  - `rowindices`: A vector where each element is a vector of row indices for the corresponding block.
  - `colindices`: A vector where each element is a vector of column indices for the corresponding block.
  - `size`: A tuple representing the size of the block sparse matrix.
  - `colors`: A vector of colors, where each color is a vector of block indices that can be
    processed in parallel without race conditions.
  - `transposecolors`: A vector of colors for the transpose matrix, where each color is a
    vector of block indices that can be processed in parallel without race conditions.
  - `scheduler`: A scheduler that manages the parallel computation of matrix-vector products.
"""
struct BlockSparseMatrix{T,M,P<:Integer,S} <: AbstractBlockMatrix{T}
    blocks::Vector{M}
    rowindices::Vector{Vector{P}}
    colindices::Vector{Vector{P}}
    size::Tuple{Int,Int}
    colors::Vector{Vector{Int}}
    transposecolors::Vector{Vector{Int}}
    scheduler::S
end

"""
    BlockSparseMatrix(
        blocks,
        rowindices,
        colindices,
        size::Tuple{Int,Int};
        coloringalgorithm=coloringalgorithm,
        scheduler=SerialScheduler(),
    )

Constructs a new `BlockSparseMatrix` instance from the given blocks, their indices, and size.

# Arguments

  - `blocks`: A vector of matrices representing the blocks.
  - `rowindices`: A vector where each element is a vector of row indices for the corresponding block.
  - `colindices`: A vector where each element is a vector of column indices for the corresponding block.
  - `size`: A tuple representing the size of the block sparse matrix.
  - `coloringalgorithm`: The algorithm from `GraphsColoring.jl` used to color the blocks for
    parallel computation. Defaults to `coloringalgorithm`.
  - `scheduler`: The scheduler used to manage parallel computation. Defaults to `SerialScheduler()`.

# Returns

  - A new `BlockSparseMatrix` instance constructed from the given blocks and size.
"""
function BlockSparseMatrix(
    blocks,
    rowindices,
    colindices,
    size::Tuple{Int,Int};
    coloringalgorithm=coloringalgorithm,
    scheduler=SerialScheduler(),
)
    return BlockSparseMatrix(
        blocks,
        rowindices,
        colindices,
        size[1],
        size[2];
        coloringalgorithm=coloringalgorithm,
        scheduler=scheduler,
    )
end

function BlockSparseMatrix(
    blocks,
    rowindices,
    colindices,
    rows::Int,
    cols::Int;
    scheduler=SerialScheduler(),
    coloringalgorithm=coloringalgorithm,
)
    # no coloring needed for single-threaded execution
    colors, transposecolors = if isserial(scheduler)
        [eachindex(blocks)], [eachindex(blocks)]
    else
        colors =
            color(conflictgraph(ColorInfo(rowindices)); algorithm=coloringalgorithm).colors
        transposecolors =
            color(conflictgraph(ColorInfo(colindices)); algorithm=coloringalgorithm).colors
        colors, transposecolors
    end

    return BlockSparseMatrix{
        eltype(eltype(typeof(blocks))),
        eltype(blocks),
        eltype(eltype(rowindices)),
        typeof(scheduler),
    }(
        blocks, rowindices, colindices, (rows, cols), colors, transposecolors, scheduler
    )
end

"""
    eachblockindex(A::BlockSparseMatrix)

Returns an iterator over the indices of the blocks in the given `BlockSparseMatrix` instance.

# Arguments

  - `A`: The `BlockSparseMatrix` instance to query.

# Returns

  - An iterator that yields the indices of the blocks in the `BlockSparseMatrix`.
"""
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

"""
    block(A::BlockSparseMatrix, i)

Returns the `i`-th block of the given `BlockSparseMatrix` instance.

# Arguments

  - `A`: The `BlockSparseMatrix` instance to query.
  - `i`: The index of the block to retrieve.

# Returns

  - The `i`-th block of the `BlockSparseMatrix`.
"""
function block(A::BlockSparseMatrix, i)
    return A.blocks[i]
end

function block(A::M, i) where {Z<:BlockSparseMatrix,T,M<:LinearMaps.AdjointMap{T,Z}}
    return adjoint(block(A.lmap, i))
end

function block(A::M, i) where {Z<:BlockSparseMatrix,T,M<:LinearMaps.TransposeMap{T,Z}}
    return transpose(block(A.lmap, i))
end

"""
    colors(A::BlockSparseMatrix)

Returns the colors used for multithreading in matrix-vector for the given
`BlockSparseMatrix`. These colors are created using `GraphsColoring.jl` and represent a
partitioning of the blocks into sets that can be processed in parallel without race conditions.

# Arguments

  - `A`: The `BlockSparseMatrix` instance to query.

# Returns

  - A collection of colors, where each color is a vector of block indices that can be processed in parallel.
"""
function colors(A::BlockSparseMatrix)
    return A.colors
end

"""
    transposecolors(A::BlockSparseMatrix)

Returns the colors used for multithreading in the transposed matrix-vector product computations for the
given `BlockSparseMatrix`. These colors are created using `GraphsColoring.jl` and represent a
partitioning of the blocks into sets that can be processed in parallel without race conditions.

# Arguments

  - `A`: The `BlockSparseMatrix` instance to query.

# Returns

  - A collection of colors, where each color is a vector of block indices that can be processed in parallel.
"""
function transposecolors(A::BlockSparseMatrix)
    return A.transposecolors
end

function colors(
    A::M
) where {
    Z<:BlockSparseMatrix,T,M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}}
}
    return transposecolors(A.lmap)
end

function SparseArrays.nnz(
    A::M
) where {
    M<:Union{
        <:BlockSparseMatrix,
        LinearMaps.AdjointMap{<:Any,<:BlockSparseMatrix},
        LinearMaps.TransposeMap{<:Any,<:BlockSparseMatrix},
    },
}
    nonzeros = 0
    for blockid in eachblockindex(A)
        nonzeros += _nnz(block(A, blockid))
    end

    return nonzeros
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector, α::Number, β::Number
) where {
    Z<:BlockSparseMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    y .*= β
    for color in colors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)

            @inbounds LinearAlgebra.mul!(
                view(y, rowindices(A, blockid)),
                block(A, blockid),
                view(x, colindices(A, blockid)),
                α,
                true,
            )
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
