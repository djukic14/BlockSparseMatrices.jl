"""
    struct ColorInfo{R}

A wrapper struct that holds conflict indices information used for graph coloring.

# Type Parameters

  - `R`: The type of the conflict indices container.

# Fields

  - `conflictindices`: A collection of indices that represent potential conflicts between blocks.
    Used to determine which blocks can be processed in parallel without race conditions.
"""
struct ColorInfo{R}
    conflictindices::R
end

"""
    conflicts(blocks::ColorInfo)

Computes the conflicts between blocks for the purpose of graph coloring using `GraphsColoring.jl`.
This function is used to determine the coloring of blocks for multithreading in the
matrix-vector product, ensuring that blocks with no conflicts can be processed in parallel.

# Arguments

  - `blocks`: A `ColorInfo` object containing the conflict indices information.

# Returns

  - A tuple containing:

      + An iterator over block indices
      + A `ConflictFunctor` wrapping the computed conflict indices
      + A range representing the maximum conflict index

# Notes

  - Blocks with no conflicts (i.e., blocks that do not overlap in their conflict indices)
    can be processed in parallel.
  - The colors are used to group blocks into sets that can be processed in parallel,
    avoiding race conditions and ensuring efficient multithreading in the matrix-vector product.
"""
function conflicts(blocks::ColorInfo)
    # indices = blocks.conflictindices
    # _conflictindices = Vector{Int}[Int[] for _ in eachindex(indices)]

    # maxconflict = 0

    # for i in eachindex(indices)
    #     _conflictindices[i] = indices[i]
    #     maxconflict = max(maxconflict, maximum(_conflictindices[i]))
    # end

    # return eachindex(indices), ConflictFunctor(_conflictindices), Base.OneTo(maxconflict)

    indices = blocks.conflictindices
    maxconflict = maximum(maximum, indices)
    return eachindex(indices), ConflictFunctor(indices), Base.OneTo(maxconflict)
end
