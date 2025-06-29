abstract type AbstractMatrixBlock{T} <: LinearMap{T} end

function Base.eltype(block::AbstractMatrixBlock{T}) where {T}
    return eltype(matrix(block))
end

function Base.eltype(::Type{<:AbstractMatrixBlock{T}}) where {T}
    return T
end

function Base.eltype(
    ::Union{Type{<:LinearMaps.AdjointMap{T,B}},Type{<:LinearMaps.TransposeMap{T,B}}}
) where {T,B<:AbstractMatrixBlock{T}}
    return T
end

function rowindices(block::AbstractMatrixBlock)
    return block.rowindices
end

function colindices(block::AbstractMatrixBlock)
    return block.colindices
end

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

function SparseArrays.nnz(
    block::M
) where {
    A<:AbstractMatrixBlock,
    M<:Union{A,LinearMaps.AdjointMap{<:Any,A},LinearMaps.TransposeMap{<:Any,A}},
}
    return nnz(matrix(block))
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
) where {
    T,
    A<:AbstractMatrixBlock,
    M<:Union{LinearMaps.AdjointMap{T,A},LinearMaps.TransposeMap{T,A}},
}
    return block.lmap.colindices
end

function colindices(
    block::M
) where {
    T,
    A<:AbstractMatrixBlock,
    M<:Union{LinearMaps.AdjointMap{T,A},LinearMaps.TransposeMap{T,A}},
}
    return block.lmap.rowindices
end

function matrix(block::M) where {T,M<:LinearMaps.TransposeMap{T,<:AbstractMatrixBlock}}
    return transpose(matrix(block.lmap))
end

function matrix(block::M) where {T,M<:LinearMaps.AdjointMap{T,<:AbstractMatrixBlock}}
    return adjoint(matrix(block.lmap))
end

function Base.axes(block::AbstractMatrixBlock)
    return (Base.OneTo(maximum(rowindices(block))), Base.OneTo(maximum(colindices(block))))
end

#TODO: check if this is the correct ordering or if it should be reversed -> benchmark performance
"""
    islessinordering(blocka::AbstractMatrixBlock, blockb::AbstractMatrixBlock)

Sorting rule for `AbstractMatrixBlock` objects
"""
function islessinordering(blocka::AbstractMatrixBlock, blockb::AbstractMatrixBlock)
    if maximum(rowindices(blocka)) < maximum(rowindices(blockb))
        return true
    else
        return maximum(colindices(blocka)) < maximum(colindices(blockb))
    end
end

struct isthreadsafe end
struct issymthreadsafe
    isthreadsafe::isthreadsafe
end

issymthreadsafe() = issymthreadsafe(isthreadsafe())

function (ists::issymthreadsafe)(blocka, blockb)
    return ists.isthreadsafe(blocka, blockb) &&
           ists.isthreadsafe(transpose(blocka), blockb) &&
           ists.isthreadsafe(blocka, transpose(blockb))
end

function (::isthreadsafe)(blocka, blockb)
    if size(blocka, 1) >= size(blockb, 1)
        issubset(rowindices(blockb), rowindices(blocka)) && (return false)

        if size(blocka, 2) >= size(blockb, 2)
            return !issubset(colindices(blockb), colindices(blocka))
        else
            return !issubset(colindices(blocka), colindices(blockb))
        end
    else
        issubset(rowindices(blocka), rowindices(blockb)) && (return false)

        if size(blocka, 2) >= size(blockb, 2)
            return !issubset(colindices(blockb), colindices(blocka))
        else
            return !issubset(colindices(blocka), colindices(blockb))
        end
    end
end

"""
    findcolor!(
        blockid::Int,
        threadsafecolors::AbstractArray,
        blocks::Vector{M};
        threadsafecheck=isthreadsafe(),
        color=1,
    ) where {M}

Assigns recursively threadsafe color to the dense block with index `blockid`.

# Arguments
- `blockid::Int`: Index of block.
- `threadsafecolors::AbstractArray`: Contains assigned blocks.
- `blocks::Vector{M}`: Vector with all blocks.
- `threadsafecheck=isthreadsafe()`: Threadsafe check, differs between symmetric and non-symmetric matrix.
- `color=1`: Currently tested color, increased if block does not fit in color.
"""
function findcolor!(
    blockid::Int,
    threadsafecolors::AbstractArray,
    blocks::Vector{M};
    threadsafecheck=isthreadsafe(),
    color=1,
) where {M}
    #This case should not appear, it is a rescue measure
    (length(threadsafecolors) < color) && return push!(threadsafecolors, [blockid])

    for testblockid in threadsafecolors[color]
        if threadsafecheck(blocks[testblockid], blocks[blockid])
            return push!(threadsafecolors[color], blockid)
        else
            return findcolor!(
                blockid,
                threadsafecolors,
                blocks;
                threadsafecheck=threadsafecheck,
                color=color + 1,
            )
        end
    end

    return push!(threadsafecolors[color], blockid)
end
