struct ColorInfo{R}
    conflictindices::R
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
function conflicts(blocks::ColorInfo)
    indices = blocks.conflictindices
    _conflictindices = Vector{Int}[Int[] for _ in eachindex(indices)]

    maxconflict = 0

    for i in eachindex(indices)
        _conflictindices[i] = conflictindices(indices[i])
        maxconflict = max(maxconflict, maximum(_conflictindices[i]))
    end

    return eachindex(indices), ConflictFunctor(_conflictindices), Base.OneTo(maxconflict)
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
function conflictindices(rowindices)
    return rowindices
end
