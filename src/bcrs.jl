# """
#     struct BlockCompressedRowStorage{T,M,P<:Integer} <: AbstractBlockMatrix{T}

# Block Compressed Row Storage (BCRS) format for block-sparse matrices where all blocks
# have uniform size. This format is efficient for matrices with regular block structure.

# # Type Parameters

#   - `T`: The element type of the matrix.
#   - `M`: The type of the matrix blocks (typically dense arrays).
#   - `P`: The integer type used for indexing (e.g., Int32, Int64).

# # Fields

#   - `blocks`: A vector containing all non-zero blocks stored row-wise.
#   - `rowptr`: Row pointer array where `rowptr[i]` indicates the start of row `i` in `blocks`.
#   - `colindices`: Column index for each block in `blocks`.
#   - `blocksize`: Tuple `(brows, bcols)` representing uniform block dimensions.
#   - `size`: Total matrix dimensions `(rows, cols)`.
#   - `forwardbuffer`: Buffer for forward matrix-vector products.
#   - `adjointbuffer`: Buffer for adjoint matrix-vector products.
#   - `buffer`: Underlying reusable buffer.
#   - `scheduler`: Scheduler for parallel computation.

# # Notes

#   - BCRS is most efficient when all blocks have the same dimensions.
#   - Storage is row-oriented, making row-wise operations more efficient.
#   - Blocks are stored contiguously in memory for cache efficiency.
# """
# struct BlockCompressedRowStorage{T,M,P<:Integer,S} <: AbstractBlockMatrix{T}
#     blocks::Vector{M}
#     rowptr::Vector{P}
#     colindices::Vector{P}
#     blocksize::Tuple{Int,Int}
#     size::Tuple{Int,Int}
#     forwardbuffer::Vector{T}
#     adjointbuffer::Vector{T}
#     buffer::Vector{T}
#     scheduler::S
# end

# """
#     BlockCompressedRowStorage(
#         blocks::Vector{M},
#         rowptr::Vector{P},
#         colindices::Vector{P},
#         blocksize::Tuple{Int,Int},
#         size::Tuple{Int,Int};
#         scheduler=SerialScheduler()
#     ) where {M,P<:Integer}

# Constructs a new `BlockCompressedRowStorage` instance.

# # Arguments

#   - `blocks`: Vector of non-zero block matrices, stored row-wise.
#   - `rowptr`: Row pointer array indicating block row boundaries.
#   - `colindices`: Column index for each block.
#   - `blocksize`: Uniform block dimensions `(brows, bcols)`.
#   - `size`: Total matrix dimensions `(rows, cols)`.
#   - `scheduler`: Scheduler for parallel operations (default: `SerialScheduler()`).

# # Returns

#   - A new `BlockCompressedRowStorage` instance.

# # Example

# ```julia
# # Create a 6×6 matrix with 2×2 blocks
# blocks = [rand(2, 2), rand(2, 2), rand(2, 2), rand(2, 2)]
# rowptr = [1, 3, 5]  # Row 1 has 2 blocks, row 2 has 2 blocks
# colindices = [1, 2, 1, 3]  # Block positions
# A = BlockCompressedRowStorage(blocks, rowptr, colindices, (2, 2), (6, 6))
# ```
# """
# function BlockCompressedRowStorage(
#     blocks::Vector{M},
#     rowptr::Vector{P},
#     colindices::Vector{P},
#     blocksize::Tuple{Int,Int},
#     size::Tuple{Int,Int};
#     scheduler=SerialScheduler(),
# ) where {M,P<:Integer}
#     T = eltype(M)
#     forwardbuffer, adjointbuffer, buffer = buffers(T, size[1], size[2])

#     return BlockCompressedRowStorage{T,M,P,typeof(scheduler)}(
#         blocks,
#         rowptr,
#         colindices,
#         blocksize,
#         size,
#         forwardbuffer,
#         adjointbuffer,
#         buffer,
#         scheduler,
#     )
# end

# """
#     BlockCompressedRowStorage(
#         bsm::BlockSparseMatrix,
#         blocksize::Tuple{Int,Int}
#     )

# Convert a `BlockSparseMatrix` to `BlockCompressedRowStorage` format.

# # Arguments

#   - `bsm`: Source `BlockSparseMatrix` to convert.
#   - `blocksize`: Uniform block dimensions `(brows, bcols)`.

# # Returns

#   - A new `BlockCompressedRowStorage` instance with the same data.

# # Notes

#   - All blocks in `bsm` must have dimensions matching `blocksize`.
#   - Throws an error if block dimensions are inconsistent.
# """
# function BlockCompressedRowStorage(
#     bsm::BlockSparseMatrix{T,M}, blocksize::Tuple{Int,Int}
# ) where {T,M}
#     nblockrows = div(size(bsm, 1), blocksize[1])
#     nblockrows * blocksize[1] == size(bsm, 1) ||
#         error("Matrix rows $(size(bsm,1)) not divisible by block rows $(blocksize[1])")

#     # Build rowptr and collect blocks
#     rowptr = Vector{Int}(undef, nblockrows + 1)
#     blocks = Matrix{T}[]
#     colindices = Int[]

#     rowptr[1] = 1
#     for brow in 1:nblockrows
#         row_start = (brow - 1) * blocksize[1] + 1
#         row_end = brow * blocksize[1]

#         nblocks_in_row = 0
#         for block in bsm.blocks
#             if row_start in rowindices(block)
#                 # Verify block size
#                 size(block.matrix) == blocksize || error(
#                     "Block size mismatch: expected $blocksize, got $(size(block.matrix))",
#                 )

#                 push!(blocks, block.matrix)
#                 bcol =
#                     div(first(BlockSparseMatrices.colindices(block)) - 1, blocksize[2]) + 1
#                 push!(colindices, bcol)
#                 nblocks_in_row += 1
#             end
#         end

#         rowptr[brow + 1] = rowptr[brow] + nblocks_in_row
#     end

#     return BlockCompressedRowStorage(
#         blocks, rowptr, colindices, blocksize, size(bsm); scheduler=scheduler(bsm)
#     )
# end

# function Base.size(A::BlockCompressedRowStorage)
#     return A.size
# end

# function scheduler(A::BlockCompressedRowStorage)
#     return A.scheduler
# end

# """
#     Base.getindex(A::BlockCompressedRowStorage, i::Integer, j::Integer)

# Get element at position `(i,j)` in the BCRS matrix.
# """
# function Base.getindex(A::BlockCompressedRowStorage, i::Integer, j::Integer)
#     (i > size(A, 1) || j > size(A, 2)) && throw(BoundsError(A, (i, j)))

#     brows, bcols = A.blocksize
#     brow = div(i - 1, brows) + 1
#     bcol = div(j - 1, bcols) + 1

#     local_i = mod1(i, brows)
#     local_j = mod1(j, bcols)

#     # Search for block in row brow with column bcol
#     for idx in A.rowptr[brow]:(A.rowptr[brow + 1] - 1)
#         if A.colindices[idx] == bcol
#             return A.blocks[idx][local_i, local_j]
#         end
#     end

#     return zero(eltype(A))
# end

# """
#     LinearAlgebra.mul!(y::AbstractVector, A::BlockCompressedRowStorage, x::AbstractVector)

# Compute matrix-vector product `y = A * x` for BCRS matrix.
# """
# function LinearAlgebra.mul!(
#     y::AbstractVector, A::BlockCompressedRowStorage, x::AbstractVector
# )
#     fill!(y, zero(eltype(y)))
#     return mul!(y, A, x, true, true)
# end

# function LinearAlgebra.mul!(
#     y::AbstractVector, A::BlockCompressedRowStorage, x::AbstractVector, α::Number, β::Number
# )
#     brows, bcols = A.blocksize
#     nblockrows = length(A.rowptr) - 1

#     y .*= β

#     for brow in 1:nblockrows
#         row_start = (brow - 1) * brows + 1
#         row_end = brow * brows

#         for bidx in A.rowptr[brow]:(A.rowptr[brow + 1] - 1)
#             bcol = A.colindices[bidx]
#             col_start = (bcol - 1) * bcols + 1
#             col_end = bcol * bcols

#             block = A.blocks[bidx]
#             x_block = @view x[col_start:col_end]
#             y_block = @view y[row_start:row_end]

#             LinearAlgebra.mul!(y_block, block, x_block, α, true)
#         end
#     end

#     return y
# end

# """
#     SparseArrays.nnz(A::BlockCompressedRowStorage)

# Returns the number of non-zero elements (including explicit zeros in stored blocks).
# """
# function SparseArrays.nnz(A::BlockCompressedRowStorage)
#     brows, bcols = A.blocksize
#     return length(A.blocks) * brows * bcols
# end
