"""
    struct VariableBlockCompressedRowStorage{T,M,P,S} <: AbstractBlockMatrix{T}

A compressed row storage format for block sparse matrices with variable-sized blocks.
This format stores blocks in row-major order, enabling efficient matrix-vector products
and parallel computation.

# Type Parameters

  - `T`: The element type of the matrix.
  - `M`: The type of the matrix blocks.
  - `P`: The integer type used for indexing.
  - `S`: The type of the scheduler.

# Fields

  - `blocks`: A vector of matrix blocks stored in row-major order.
  - `rowptr`: A vector of pointers indicating the start of each block row in the `blocks` vector.
    Similar to CSR format, `rowptr[i]` points to the first block of the i-th block row.
  - `colindices`: A vector where each element is the starting column index of the corresponding block.
    The indices for each block must be contiguous (e.g., if a block starts at column 5 and has 3 columns,
    it occupies columns 5, 6, and 7).
  - `rowindices`: A vector where each element is the starting row index of the corresponding block.
    The indices for each block must be contiguous (e.g., if a block starts at row 10 and has 4 rows,
    it occupies rows 10, 11, 12, and 13).
  - `size`: A tuple representing the total size of the matrix.
  - `scheduler`: A scheduler that manages the parallel computation of matrix-vector products.

# Notes

  - Blocks are sorted by row index first, then by column index within each row.
  - The compressed row storage format allows efficient parallel computation across block rows.
  - This format is particularly efficient for matrix-vector products and can handle variable-sized blocks.
  - Each block must occupy a contiguous range of row and column indices.
"""
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
        matrices,
        rowindices,
        colindices,
        matrixsize::Tuple{Int,Int};
        scheduler = SerialScheduler()
    )

Constructs a `VariableBlockCompressedRowStorage` from vectors of matrices and their corresponding
row and column indices. The blocks are automatically sorted by row index first, then by column index,
and stored in compressed row storage format.

# Arguments

  - `matrices`: A vector of matrices representing the blocks.
  - `rowindices`: A vector where each element is the starting row index of the corresponding block.
    Each block must occupy contiguous row indices starting from this value.
  - `colindices`: A vector where each element is the starting column index of the corresponding block.
    Each block must occupy contiguous column indices starting from this value.
  - `matrixsize`: The total size of the matrix as a tuple `(nrows, ncols)`.
  - `scheduler`: A scheduler for parallel computation. Defaults to `SerialScheduler()`.

# Returns

  - A `VariableBlockCompressedRowStorage` instance with compressed row storage format.

# Notes

  - Blocks do not need to be provided in sorted order; they will be sorted internally.
  - The row and column indices for each block must be contiguous.
  - Blocks with the same row index are grouped together in the compressed format.
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

"""
    VariableBlockCompressedRowStorage(
        bsm::BlockSparseMatrix;
        scheduler=bsm.scheduler
    )

Converts a `BlockSparseMatrix` to `VariableBlockCompressedRowStorage` format.

# Arguments

  - `bsm`: A `BlockSparseMatrix` to convert.
  - `scheduler`: A scheduler for parallel computation. Defaults to the scheduler from `bsm`.

# Returns

  - A `VariableBlockCompressedRowStorage` instance in compressed row storage format.

# Notes

  - No sanity checks are performed on the input matrix. It is assumed that the blocks in the
    `BlockSparseMatrix` have contiguous row and column indices.
  - The conversion uses lazy functors to avoid unnecessary memory allocations during construction.
"""
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

"""
    VariableBlockCompressedRowStorage(
        sbm::SymmetricBlockMatrix;
        scheduler=sbm.scheduler
    )

Converts a `SymmetricBlockMatrix` to `VariableBlockCompressedRowStorage` format.
The symmetric structure is expanded by including both the diagonal blocks, off-diagonal blocks,
and their transposes explicitly.

# Arguments

  - `sbm`: A `SymmetricBlockMatrix` to convert.
  - `scheduler`: A scheduler for parallel computation. Defaults to the scheduler from `sbm`.

# Returns

  - A `VariableBlockCompressedRowStorage` instance in compressed row storage format.

# Notes

  - No sanity checks are performed on the input matrix. It is assumed that the blocks in the
    `SymmetricBlockMatrix` have contiguous row and column indices.
  - The conversion expands the symmetric structure: diagonal blocks are included once, while
    off-diagonal blocks are included twice (once as-is and once transposed).
  - The conversion uses lazy functors to avoid unnecessary memory allocations during construction.
"""
function VariableBlockCompressedRowStorage(
    sbm::SymmetricBlockMatrix{T,D,P,M}; scheduler=sbm.scheduler
) where {T,D,P,M}
    return VariableBlockCompressedRowStorage(
        _SymmetricMatrixFunctor(sbm),
        _SymmetricRowIndexFunctor(sbm),
        _SymmetricColIndexFunctor(sbm),
        size(sbm);
        scheduler=scheduler,
    )
end

for (FunctorName, accessor, lengthfield) in [
    (:_MatrixFunctor, :(f.b.blocks[i]), :blocks),
    (:_RowIndexFunctor, :(first(rowindices(f.b, i))), :blocks),
    (:_ColIndexFunctor, :(first(colindices(f.b, i))), :blocks),
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

# Functors for SymmetricBlockMatrix using metaprogramming
for (FunctorName, diag_accessor, offdiag_accessor, transpose_offdiag_accessor) in [
    (
        :_SymmetricMatrixFunctor,
        :(f.b.diagonals[i]),
        :(f.b.offdiagonals[i - ndiag]),
        :(transpose(f.b.offdiagonals[i - ndiag - noffdiag])),
    ),
    (
        :_SymmetricRowIndexFunctor,
        :(first(f.b.diagonalindices[i])),
        :(first(f.b.rowindices[i - ndiag])),
        :(first(f.b.colindices[i - ndiag - noffdiag])),
    ),
    (
        :_SymmetricColIndexFunctor,
        :(first(f.b.diagonalindices[i])),
        :(first(f.b.colindices[i - ndiag])),
        :(first(f.b.rowindices[i - ndiag - noffdiag])),
    ),
]
    @eval begin
        struct $FunctorName{B}
            b::B
        end

        function Base.getindex(f::$FunctorName, i::Int)
            ndiag = length(f.b.diagonals)
            noffdiag = length(f.b.offdiagonals)

            if i <= ndiag
                return $diag_accessor
            elseif i <= ndiag + noffdiag
                return $offdiag_accessor
            else
                return $transpose_offdiag_accessor
            end
        end

        function Base.length(f::$FunctorName)
            return length(f.b.diagonals) + 2 * length(f.b.offdiagonals)
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
