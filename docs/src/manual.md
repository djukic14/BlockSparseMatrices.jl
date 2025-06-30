
# General Usage

The basic usage of this package is provided in the following examples.

## Example: Block Sparse Matrix
```@example blocksparsematrix
using BlockSparseMatrices

# define dense matrix blocks
mat1 = randn(ComplexF64, 2, 2)
block1 = BlockSparseMatrices.DenseMatrixBlock(mat1, 1:2, 1:2)
mat2 = randn(ComplexF64, 3, 3)
block2 = BlockSparseMatrices.DenseMatrixBlock(mat2, 3:5, 3:5)

blockmatrix = BlockSparseMatrix([block1, block2], 5, 5)
```

## Example: Symmetric Block Sparse Matrix
```@example symblocksparsematrix
using BlockSparseMatrices

# define dense matrix blocks
mat1 = randn(ComplexF64, 2, 2)
diagonalblock = BlockSparseMatrices.DenseMatrixBlock(mat1, 1:2, 1:2)
mat2 = randn(ComplexF64, 3, 3)
offdiagonalblock = BlockSparseMatrices.DenseMatrixBlock(mat2, 7:9, 7:9)

diagonalmatrix = SymmetricBlockMatrix([diagonalblock], [offdiagonalblock], 9, 9)
```