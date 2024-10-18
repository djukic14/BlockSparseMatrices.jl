using LinearAlgebra
using Test
using BlockSparseMatrices
using SparseArrays

mat1 = randn(ComplexF64, 2, 2)
mat2 = randn(ComplexF64, 3, 3)
mat3 = randn(ComplexF64, 2, 3)
mat4 = Matrix(transpose(mat3))

M = [mat1 mat3; transpose(mat3) mat2]

block1 = BlockSparseMatrices.DenseMatrixBlock(mat1, 1:2, 1:2)
block2 = BlockSparseMatrices.DenseMatrixBlock(mat2, 3:5, 3:5)
block3 = BlockSparseMatrices.DenseMatrixBlock(mat3, 1:2, 3:5)
block4 = BlockSparseMatrices.DenseMatrixBlock(mat4, 3:5, 1:2)

blockmatrix = BlockSparseMatrix([block1, block2, block3, block4], 5, 5)
blockmatrix2 = BlockSparseMatrix([block1, block2, block3, block4], (5, 5))
blockmatrix3 = BlockSparseMatrix(
    [mat1, mat2, mat3, mat4], [1:2, 3:5, 1:2, 3:5], [1:2, 3:5, 3:5, 1:2], (5, 5)
)

@test blockmatrix.blocks == blockmatrix2.blocks == blockmatrix3.blocks

@test size(blockmatrix) == (5, 5)
@test size(blockmatrix, 1) == size(blockmatrix, 2) == 5
@test size(transpose(blockmatrix)) == (5, 5)
@test size(transpose(blockmatrix), 1) == size(transpose(blockmatrix), 2) == 5
@test size(adjoint(blockmatrix)) == (5, 5)
@test size(adjoint(blockmatrix), 1) == size(adjoint(blockmatrix), 2) == 5
@test nnz(blockmatrix) == 25

x = rand(ComplexF64, 5)
@test blockmatrix * x ≈ M * x
@test adjoint(blockmatrix) * x ≈ adjoint(M) * x
@test transpose(blockmatrix) * x ≈ transpose(M) * x

x1 = randn(ComplexF64, 5)
x2 = deepcopy(x1)

α = randn(ComplexF64)
β = randn(ComplexF64)

LinearAlgebra.mul!(x1, blockmatrix, x, α, β)
LinearAlgebra.mul!(x2, M, x, α, β)
@test x1 ≈ x2
LinearAlgebra.mul!(x1, adjoint(blockmatrix), x, α, β)
LinearAlgebra.mul!(x2, adjoint(M), x, α, β)
@test x1 ≈ x2
LinearAlgebra.mul!(x1, transpose(blockmatrix), x, α, β)
LinearAlgebra.mul!(x2, transpose(M), x, α, β)
@test x1 ≈ x2
