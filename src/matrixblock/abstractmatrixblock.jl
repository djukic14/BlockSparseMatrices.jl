"""
    abstract type AbstractMatrixBlock{T} <: LinearMap{T}

Abstract type representing a matrix block with element type `T`. This type inherits from
`LinearMap{T}`, indicating that matrix-vector products can be performed with instances of
this type, even when considering a single block in isolation. The dimensions of the vectors
involved in the matrix-vector product are the same as the dimensions of the block.

Additionally, transpose and adjoint operations are also supported.

# Notes

  - The matrix-vector product operation is well-defined for a single block.
  - Both transpose and adjoint operations are available.
"""
abstract type AbstractMatrixBlock{T} <: LinearMap{T} end

function Base.eltype(block::AbstractMatrixBlock{T}) where {T}
    return eltype(matrix(block))
end

function Base.eltype(::Type{<:AbstractMatrixBlock{T}}) where {T}
    return T
end

"""
    rowindices(block::AbstractMatrixBlock)

Returns the global row indices of the matrix block in the sparse block matrix.

# Notes

  - The returned indices are global indices in the sparse block matrix, not local indices inside the block.
"""
function rowindices(block::AbstractMatrixBlock)
    return block.rowindices
end

"""
    colindices(block::AbstractMatrixBlock)

Returns the global column indices of the matrix block in the sparse block matrix.

# Notes

  - The returned indices are global indices in the sparse block matrix, not local indices inside the block.
"""
function colindices(block::AbstractMatrixBlock)
    return block.colindices
end

"""
    matrix(block::AbstractMatrixBlock)

Returns the content of the matrix block.

# Returns

  - `matrix`: The actual matrix stored in the block.

# Notes

  - This function provides direct access to the matrix data, allowing for further manipulation.
"""
function matrix(block::AbstractMatrixBlock)
    return block.matrix
end

function Base.size(
    block::M
) where {
    A<:AbstractMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{<:Any,A},LinearMaps.TransposeMap{<:Any,A}},
}
    return (length(rowindices(block)), length(colindices(block)))
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, block::M, x::AbstractVector
) where {
    A<:AbstractMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{<:Any,A},LinearMaps.TransposeMap{<:Any,A}},
}
    LinearAlgebra.mul!(y, matrix(block), x, true, true)

    return y
end

function rowindices(
    block::M
) where {T,A<:Any,M<:Union{LinearMaps.AdjointMap{T,A},LinearMaps.TransposeMap{T,A}}}
    return colindices(block.lmap)
end

function colindices(
    block::M
) where {T,A<:Any,M<:Union{LinearMaps.AdjointMap{T,A},LinearMaps.TransposeMap{T,A}}}
    return rowindices(block.lmap)
end

function matrix(block::M) where {T,M<:LinearMaps.TransposeMap{T,<:Any}}
    return transpose(matrix(block.lmap))
end

function matrix(block::M) where {T,M<:LinearMaps.AdjointMap{T,<:Any}}
    return adjoint(matrix(block.lmap))
end

"""
    islessinordering(blocka::AbstractMatrixBlock, blockb::AbstractMatrixBlock)

Defines a sorting rule for `AbstractMatrixBlock` objects. This function determines the order
of two blocks based on their row and column indices.

# Returns

  - `true` if `blocka` is considered less than `blockb` according to the sorting rule,
    `false` otherwise.

# Sorting Rule

  - Blocks are first compared based on the maximum row index. If the maximum row index of
    `blocka` is less than that of `blockb`, `blocka` is considered less than `blockb`.
  - If the maximum row indices are equal, the blocks are compared based on the maximum column
    index. If the maximum column index of `blocka` is less than that of `blockb`, `blocka` is
    considered less than `blockb`.
"""
function islessinordering(blocka::AbstractMatrixBlock, blockb::AbstractMatrixBlock)
    if maximum(rowindices(blocka)) < maximum(rowindices(blockb))
        return true
    else
        return maximum(colindices(blocka)) < maximum(colindices(blockb))
    end
end

"""
    conflicts(blocks::Vector{A}; kwargs...) where {A<:AbstractMatrixBlock}

Computes the conflicts between blocks for the purpose of graph coloring using `GraphsColoring.jl`.
This function is used to determine the coloring of blocks for multithreading in the
matrix-vector product, ensuring that blocks with no conflicts can be processed in parallel.

# Arguments

  - `blocks`: A vector of `AbstractMatrixBlock` objects.
  - `kwargs...`: Additional keyword arguments passed to the `conflictindices` function.

# Notes

  - Blocks with no conflicts (i.e., blocks that do not overlap in their conflict indices)
    can be processed in parallel.
  - The colors are used to group blocks into sets that can be processed in parallel,
    avoiding race conditions and ensuring efficient multithreading in the matrix-vector product.
"""
function conflicts(blocks::Vector{A}; kwargs...) where {A<:AbstractMatrixBlock}
    _conflictindices = Vector{Int}[Int[] for _ in eachindex(blocks)]

    maxconflict = 0

    for i in eachindex(blocks)
        _conflictindices[i] = conflictindices(blocks[i]; kwargs...)
        maxconflict = max(maxconflict, maximum(_conflictindices[i]))
    end

    return eachindex(blocks), ConflictFunctor(_conflictindices), Base.OneTo(maxconflict)
end

"""
    conflictindices(block::AbstractMatrixBlock; transpose=false)

Returns the conflict indices for a given `AbstractMatrixBlock` object.
These indices represent the memory locations that are accessed by the block during a
matrix-vector product.

# Arguments

  - `block`: The `AbstractMatrixBlock` object for which to compute the conflict indices.
  - `transpose`: A boolean flag indicating whether to consider the transpose of the block.
    Defaults to `false`.

# Returns

  - A collection of indices representing the memory locations that are accessed by the block.

# Notes

  - If `transpose` is `false`, the function returns the row indices of the block, as these
    correspond to the memory locations accessed during a standard matrix-vector product.
  - If `transpose` is `true`, the function returns the column indices of the block, as these
    correspond to the memory locations accessed during a transposed (and adjoint)
    matrix-vector product.
"""
function conflictindices(block::AbstractMatrixBlock; transpose=false)
    transpose && return colindices(block)
    return rowindices(block)
end
