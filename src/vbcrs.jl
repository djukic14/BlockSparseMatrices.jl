struct VariableBlockCompressedRowStorage{T,M,P<:Integer,S} <: AbstractBlockMatrix{T}
    blocks::Vector{M}
    rowptr::Vector{P}
    colindices::Vector{P}
    rowindices::Vector{P}
    size::Tuple{Int,Int}
    scheduler::S
end

"""
    VariableBlockCompressedRowStorage(
        matrices::Vector{M},
        rowindices::Vector{V},
        colindices::Vector{V},
        matrixsize::Tuple{Int,Int};
        scheduler = SerialScheduler()
    ) where {M,V}

Constructs a `VariableBlockCompressedRowStorage` from vectors of matrices and their corresponding
row and column indices.

# Arguments

  - `matrices`: A vector of matrices representing the blocks.
  - `rowindices`: A vector where each element is the starting row index of the corresponding block.
  - `colindices`: A vector where each element is the starting column index of the corresponding block.
  - `matrixsize`: The total size of the matrix as a tuple `(nrows, ncols)`.
  - `scheduler`: A scheduler for parallel computation.

# Returns

  - A `VariableBlockCompressedRowStorage` instance with compressed row storage format.
"""
function VariableBlockCompressedRowStorage(
    matrices, rowindices, colindices, matrixsize; scheduler=SerialScheduler()
)
    M = typeof(matrices[1])
    V = typeof(rowindices[1])

    perm = sortperm(1:length(matrices); by=i -> (rowindices[i], colindices[i]))

    # Count unique block rows and build rowptr in one pass
    nblockrows = 0
    prevrow = typemin(V)
    for i in perm
        if rowindices[i] != prevrow
            nblockrows += 1
            prevrow = rowindices[i]
        end
    end

    # Pre-allocate output arrays
    rowptr = Vector{Int}(undef, nblockrows + 1)
    blocks = Vector{M}(undef, length(matrices))
    cindices = Vector{V}(undef, length(matrices))
    rowindicesvec = Vector{V}(undef, length(matrices))

    # Fill the arrays in sorted order
    rowptr[1] = 1
    rowidx = 1
    prevrow = rowindices[perm[1]]

    for (outidx, inidx) in enumerate(perm)
        currrow = rowindices[inidx]
        if currrow != prevrow
            rowptr[rowidx + 1] = outidx
            rowidx += 1
            prevrow = currrow
        end

        blocks[outidx] = matrices[inidx]
        cindices[outidx] = colindices[inidx]
        rowindicesvec[outidx] = rowindices[inidx]
    end
    rowptr[end] = length(matrices) + 1

    return VariableBlockCompressedRowStorage{eltype(M),M,V,typeof(scheduler)}(
        blocks, rowptr, cindices, rowindicesvec, matrixsize, scheduler
    )
end

#TODO: add a second ordering for transpose/adjoint MVP?
#TODO: add symmetric version of this

function VariableBlockCompressedRowStorage(
    bsm::BlockSparseMatrix{T,M}; scheduler=bsm.scheduler
) where {T,M}
    return VariableBlockCompressedRowStorage(
        _MatrixFunctor(bsm),
        _RowIndexFunctor(bsm),
        _ColIndexFunctor(bsm),
        size(bsm);
        scheduler=scheduler,
    )
end

for (FunctorName, accessor, lengthfield) in [
    (:_MatrixFunctor, :(matrix(f.b.blocks[i])), :blocks),
    (:_RowIndexFunctor, :(first(rowindices(f.b.blocks[i]))), :blocks),
    (:_ColIndexFunctor, :(first(colindices(f.b.blocks[i]))), :blocks),
]
    @eval begin
        struct $FunctorName{B}
            b::B
        end

        function Base.getindex(f::$FunctorName, i::Int)
            return $accessor
        end

        function Base.length(f::$FunctorName)
            return length(f.b.$lengthfield)
        end
    end
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    A::VariableBlockCompressedRowStorage,
    x::AbstractVector,
    α::Number,
    β::Number,
)
    y .*= β

    @tasks for browidx in 1:(length(A.rowptr) - 1)
        @set scheduler = BlockSparseMatrices.scheduler(A)
        for bidx in A.rowptr[browidx]:(A.rowptr[browidx + 1] - 1)
            block = A.blocks[bidx]
            x_block = view(x, A.colindices[bidx]:(A.colindices[bidx] + size(block, 2) - 1))
            y_block = view(y, A.rowindices[bidx]:(A.rowindices[bidx] + size(block, 1) - 1))
            LinearAlgebra.mul!(y_block, block, x_block, α, true)
        end
    end

    return y
end

function SparseArrays.nnz(A::VariableBlockCompressedRowStorage)
    total = 0
    for block in A.blocks
        total += length(block)
    end
    return total
end

# Dispatch on map type to get the appropriate operation
_block_op(::Type{<:LinearMaps.AdjointMap}) = adjoint
_block_op(::Type{<:LinearMaps.TransposeMap}) = transpose

# Helper function for adjoint/transpose multiplication
function _unsafe_mul_transpose!(
    y::AbstractVector, A::M, x::AbstractVector, α::Number, β::Number
) where {
    M<:Union{
        LinearMaps.AdjointMap{<:Any,<:VariableBlockCompressedRowStorage},
        LinearMaps.TransposeMap{<:Any,<:VariableBlockCompressedRowStorage},
    },
}
    op = _block_op(M)
    lmap = A.lmap
    y .*= β

    for browidx in 1:(length(lmap.rowptr) - 1)
        for bidx in lmap.rowptr[browidx]:(lmap.rowptr[browidx + 1] - 1)
            block = lmap.blocks[bidx]
            y_block = view(
                y, lmap.colindices[bidx]:(lmap.colindices[bidx] + size(block, 2) - 1)
            )
            x_block = view(
                x, lmap.rowindices[bidx]:(lmap.rowindices[bidx] + size(block, 1) - 1)
            )
            LinearAlgebra.mul!(y_block, op(block), x_block, α, true)
        end
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    A::Union{
        LinearMaps.AdjointMap{<:Any,<:VariableBlockCompressedRowStorage},
        LinearMaps.TransposeMap{<:Any,<:VariableBlockCompressedRowStorage},
    },
    x::AbstractVector,
)
    fill!(y, zero(eltype(y)))
    return LinearMaps._unsafe_mul!(y, A, x, true, true)
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    A::Union{
        LinearMaps.AdjointMap{<:Any,<:VariableBlockCompressedRowStorage},
        LinearMaps.TransposeMap{<:Any,<:VariableBlockCompressedRowStorage},
    },
    x::AbstractVector,
    α::Number,
    β::Number,
)
    return _unsafe_mul_transpose!(y, A, x, α, β)
end
