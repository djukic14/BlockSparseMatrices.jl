# Compressed Row Storage Formats

BlockSparseMatrices.jl provides two compressed row storage formats optimized for different use cases:

- **Block Compressed Row Storage (BCRS)**: Efficient format for matrices with uniform block sizes
- **Variable Block Compressed Row Storage (VBCRS)**: Flexible format supporting blocks of varying dimensions

## Block Compressed Row Storage (BCRS)

The BCRS format is ideal for matrices where all blocks have the same dimensions. It provides:

- Minimal storage overhead
- Fast indexing and matrix-vector products
- Cache-efficient memory layout

### Construction

```julia
using BlockSparseMatrices

# Create a 6×6 matrix with 2×2 blocks
blocks = [
    [1.0 2.0; 3.0 4.0],  # Block (1,1)
    [5.0 6.0; 7.0 8.0],  # Block (1,2)
    [9.0 10.0; 11.0 12.0],  # Block (2,1)
    [13.0 14.0; 15.0 16.0]  # Block (2,3)
]

# Row pointer: [1, 3, 5] means row 1 has blocks 1-2, row 2 has blocks 3-4
rowptr = [1, 3, 5]

# Column indices: block positions in each row
colindices = [1, 2, 1, 3]

# Uniform block size
blocksize = (2, 2)

# Total matrix size
matsize = (4, 6)

A = BlockCompressedRowStorage(blocks, rowptr, colindices, blocksize, matsize)
```

### Conversion from BlockSparseMatrix

```julia
# Create a BlockSparseMatrix
blocks_vec = [rand(2,2), rand(2,2), rand(2,2)]
rowindices = [[1,2], [1,2], [3,4]]
colindices = [[1,2], [3,4], [1,2]]

bsm = BlockSparseMatrix(blocks_vec, rowindices, colindices, (4,4))

# Convert to BCRS (all blocks must have size (2,2))
bcrs = BlockCompressedRowStorage(bsm, (2,2))
```

### Operations

BCRS supports standard matrix operations:

```julia
# Element access
value = A[i, j]

# Matrix-vector product
x = rand(6)
y = A * x

# In-place multiplication with scaling
mul!(y, A, x, α, β)  # y = α*A*x + β*y

# Number of non-zeros
nz = nnz(A)
```

## Variable Block Compressed Row Storage (VBCRS)

The VBCRS format supports blocks of different sizes within the same matrix. It provides:

- Flexibility for irregular block structures
- Efficient storage for variable-sized blocks
- Full compatibility with BlockSparseMatrix

### Construction

```julia
# Create a matrix with variable-sized blocks
blocks = [
    [1.0 2.0; 3.0 4.0],              # 2×2 block
    [5.0 6.0 7.0; 8.0 9.0 10.0],     # 2×3 block
    [11.0 12.0; 13.0 14.0; 15.0 16.0]  # 3×2 block
]

# Row pointer for block rows
rowptr = [1, 3, 4]

# Column starting indices
colindices = [1, 3, 1]

# Row starting indices for each block
rowindices = [1, 1, 3]

# Size of each block
blocksizes = [(2,2), (2,3), (3,2)]

# Total matrix size
matsize = (5, 5)

A = VariableBlockCompressedRowStorage(
    blocks, rowptr, colindices, rowindices, blocksizes, matsize
)
```

### Conversion from BlockSparseMatrix

```julia
# Create a BlockSparseMatrix with variable-sized blocks
blocks_vec = [
    rand(2,2),   # 2×2 block
    rand(2,3),   # 2×3 block
    rand(3,2)    # 3×2 block
]
rowindices = [[1,2], [1,2], [3,4,5]]
colindices = [[1,2], [3,4,5], [1,2]]

bsm = BlockSparseMatrix(blocks_vec, rowindices, colindices, (5,5))

# Convert to VBCRS (automatically handles variable sizes)
vbcrs = VariableBlockCompressedRowStorage(bsm)
```

### Operations

VBCRS supports the same operations as BCRS:

```julia
# Element access
value = A[i, j]

# Matrix-vector product
x = rand(5)
y = A * x

# In-place multiplication
mul!(y, A, x, α, β)

```

## Performance Considerations

### When to use BCRS

- All blocks have identical dimensions
- Regular grid-like block structure
- Maximum memory efficiency and speed are priorities
- Examples: finite difference discretizations, uniform mesh applications

### When to use VBCRS

- Blocks have varying dimensions
- Irregular or adaptive mesh structures
- Flexibility is more important than minimal overhead
- Examples: adaptive mesh refinement, multi-scale problems

### Memory Layout

Both formats store blocks contiguously in row-major order for optimal cache performance:

```julia
# BCRS memory layout (uniform 2×2 blocks)
# [Block₁₁ | Block₁₂ | Block₂₁ | Block₂₂ | ...]
#   rows    rows      rows      rows

# VBCRS memory layout (variable sizes)
# [Block₁₁(2×2) | Block₁₂(2×3) | Block₂₁(3×2) | ...]
#   rows          rows            rows
```

## Parallel Operations

Both formats support parallel matrix-vector products using the scheduler system:

```julia
using OhMyThreads

# Serial execution (default)
A = BlockCompressedRowStorage(blocks, rowptr, colindices, blocksize, matsize)

# Parallel execution
A = BlockCompressedRowStorage(
    blocks, rowptr, colindices, blocksize, matsize;
    scheduler=DynamicScheduler()
)
```

## API Reference

```@docs
BlockCompressedRowStorage
VariableBlockCompressedRowStorage
```
