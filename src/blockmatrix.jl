"""
    struct BlockSparseMatrix{T,M,D,S} <: AbstractBlockMatrix{T}

A concrete implementation of a block sparse matrix, which is a sparse matrix composed of
smaller dense matrix blocks.

# Type Parameters

  - `T`: The element type of the matrix.
  - `M`: The type of the matrix blocks.
  - `D`: The type of the row and column index dictionaries.
  - `S`: The type of the scheduler.

# Fields

  - `blocks`: A vector of matrix blocks that comprise the block sparse matrix.
  - `size`: A tuple representing the size of the block sparse matrix.
  - `forwardbuffer`: A buffer used for forward matrix-vector product computations.
  - `adjointbuffer`: A buffer used for adjoint matrix-vector product computations.
  - `buffer`: The underlying buffer that is reused for both forward and adjoint products.
  - `rowindexdict`: A dictionary that maps row indices to block indices.
  - `colindexdict`: A dictionary that maps column indices to block indices.
  - `colors`: A vector of colors, where each color is a vector of block indices that can be
    processed in parallel without race conditions.
  - `transposecolors`: A vector of colors for the transpose matrix, where each color is a
    vector of block indices that can be processed in parallel without race conditions.
  - `scheduler`: A scheduler that manages the parallel computation of matrix-vector products.
"""
struct BlockSparseMatrix{T,M,D,S} <: AbstractBlockMatrix{T}
    blocks::Vector{M}
    size::Tuple{Int,Int}
    forwardbuffer::Vector{T}
    adjointbuffer::Vector{T}
    buffer::Vector{T}
    rowindexdict::D
    colindexdict::D #TODO: find smarter way to search for entries in matrix
    colors::Vector{Vector{Int}}
    transposecolors::Vector{Vector{Int}}
    scheduler::S
end

"""
    BlockSparseMatrix(
        blocks::Vector{M},
        rowindices::V,
        colindices::V,
        size::Tuple{Int,Int};
        coloringalgorithm=coloringalgorithm,
        scheduler=DynamicScheduler(),
    ) where {M,V}

Constructs a new `BlockSparseMatrix` instance from the given blocks, row indices, column
indices, and size.

# Arguments

  - `blocks`: A vector of dense matrices.
  - `rowindices`: A vector of row indices corresponding to each block.
  - `colindices`: A vector of column indices corresponding to each block.
  - `size`: A tuple representing the size of the block sparse matrix.
  - `coloringalgorithm`: The algorithm from `GraphColoring.jl` used to color the blocks for
    parallel computation. Defaults to `coloringalgorithm`.
  - `scheduler`: The scheduler used to manage parallel computation. Defaults to `SerialScheduler()`.

# Returns

  - A new `BlockSparseMatrix` instance constructed from the given blocks, row indices, column indices, and size.
"""
function BlockSparseMatrix(
    blocks::Vector{M},
    rowindices::V,
    colindices::V,
    size::Tuple{Int,Int};
    coloringalgorithm=coloringalgorithm,
    scheduler=SerialScheduler(),
) where {M,V}
    return BlockSparseMatrix(
        denseblocks(blocks, rowindices, colindices),
        size;
        coloringalgorithm=coloringalgorithm,
        scheduler=scheduler,
    )
end

"""
    BlockSparseMatrix(
        blocks::Vector{M},
        size::Tuple{Int,Int};
        coloringalgorithm=coloringalgorithm,
        scheduler=SerialScheduler(),
    ) where {M<:AbstractMatrixBlock}

Constructs a new `BlockSparseMatrix` instance from the given blocks and size.

# Arguments

  - `blocks`: A vector of `AbstractMatrixBlock` instances.
  - `size`: A tuple representing the size of the block sparse matrix.
  - `coloringalgorithm`: The algorithm from `GraphColoring.jl` used to color the blocks for
    parallel computation. Defaults to `coloringalgorithm`.
  - `scheduler`: The scheduler used to manage parallel computation. Defaults to `SerialScheduler()`.

# Returns

  - A new `BlockSparseMatrix` instance constructed from the given blocks and size.
"""
function BlockSparseMatrix(
    blocks::Vector{M},
    size::Tuple{Int,Int};
    coloringalgorithm=coloringalgorithm,
    scheduler=SerialScheduler(),
) where {M<:AbstractMatrixBlock}
    return BlockSparseMatrix(
        blocks, size[1], size[2]; coloringalgorithm=coloringalgorithm, scheduler=scheduler
    )
end

function BlockSparseMatrix(
    blocks::Vector{M},
    rows::Int,
    cols::Int;
    scheduler=SerialScheduler(),
    coloringalgorithm=coloringalgorithm,
) where {M<:AbstractMatrixBlock}
    forwardbuffer, adjointbuffer, buffer = buffers(eltype(M), rows, cols)

    sort!(blocks; lt=islessinordering)

    rowindexdict = Dict{Int,Vector{Int}}()
    colindexdict = Dict{Int,Vector{Int}}()

    for (i, block) in enumerate(blocks)
        _appendindexdict!(rowindexdict, block.rowindices, i)
        _appendindexdict!(colindexdict, block.colindices, i)
    end

    # no coloring needed for single-threaded execution
    colors, transposecolors = if isserial(scheduler)
        [eachindex(blocks)], [eachindex(blocks)]
    else
        colors = color(conflictgraph(blocks; transpose=false); algorithm=coloringalgorithm)
        transposecolors = color(
            conflictgraph(blocks; transpose=true); algorithm=coloringalgorithm
        )
        colors, transposecolors
    end

    return BlockSparseMatrix{eltype(M),M,typeof(rowindexdict),typeof(scheduler)}(
        blocks,
        (rows, cols),
        forwardbuffer,
        adjointbuffer,
        buffer,
        rowindexdict,
        colindexdict,
        colors,
        transposecolors,
        scheduler,
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
`BlockSparseMatrix`. These colors are created using `GraphColoring.jl` and represent a
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
    colors(A::BlockSparseMatrix)

Returns the colors used for multithreading in the transposed matrix-vector product computations for the
given `BlockSparseMatrix`. These colors are created using `GraphColoring.jl` and represent a
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
    for color in colors(A)
        @tasks for blockid in color
            @set scheduler = BlockSparseMatrices.scheduler(A)

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
