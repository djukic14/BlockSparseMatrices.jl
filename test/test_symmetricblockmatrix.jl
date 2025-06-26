using LinearAlgebra
using Test
using BlockSparseMatrices
using SparseArrays

block1 = randn(ComplexF64, 2, 2)
block2 = randn(ComplexF64, 3, 3)
block3 = randn(ComplexF64, 2, 3)

M = [block1 block3; transpose(block3) block2]

diagonal1 = BlockSparseMatrices.DenseMatrixBlock(block1, 1:2, 1:2)
diagonal2 = BlockSparseMatrices.DenseMatrixBlock(block2, 3:5, 3:5)
offdiagonal1 = BlockSparseMatrices.DenseMatrixBlock(block3, 1:2, 3:5)

diagonalmatrix = SymmetricBlockMatrix([diagonal1, diagonal2], [offdiagonal1], 5, 5)
diagonalmatrix2 = SymmetricBlockMatrix([diagonal1, diagonal2], [offdiagonal1], (5, 5))
diagonalmatrix3 = SymmetricBlockMatrix(
    [block1, block2], [1:2, 3:5], [1:2, 3:5], [block3], [1:2], [3:5], (5, 5)
)
@test diagonalmatrix.offdiagonals ==
    diagonalmatrix2.offdiagonals ==
    diagonalmatrix3.offdiagonals
@test diagonalmatrix.diagonals == diagonalmatrix2.diagonals == diagonalmatrix3.diagonals

@test size(diagonalmatrix) == (5, 5)
@test size(diagonalmatrix, 1) == size(diagonalmatrix, 2) == 5
@test size(transpose(diagonalmatrix)) == (5, 5)
@test size(transpose(diagonalmatrix), 1) == size(transpose(diagonalmatrix), 2) == 5
@test size(adjoint(diagonalmatrix)) == (5, 5)
@test size(adjoint(diagonalmatrix), 1) == size(adjoint(diagonalmatrix), 2) == 5
@test nnz(diagonalmatrix) == 25
@test diagonalmatrix[1, 1] == block1[1, 1]
@test diagonalmatrix[3, 3] == block2[1, 1]
@test diagonalmatrix[1, 3] == block3[1, 1]
@test diagonalmatrix[3, 1] == block3[1, 1]
@test BlockSparseMatrices.eachoffdiagonalindex(diagonalmatrix) ==
    BlockSparseMatrices.eachoffdiagonalindex(transpose(diagonalmatrix))

x = rand(ComplexF64, 5)
@test diagonalmatrix * x ≈ M * x
@test adjoint(diagonalmatrix) * x ≈ adjoint(M) * x
@test transpose(diagonalmatrix) * x ≈ transpose(M) * x

x1 = randn(ComplexF64, 5)
x2 = deepcopy(x1)

α = randn(ComplexF64)
β = randn(ComplexF64)

LinearAlgebra.mul!(x1, diagonalmatrix, x, α, β)
LinearAlgebra.mul!(x2, M, x, α, β)
@test x1 ≈ x2
LinearAlgebra.mul!(x1, adjoint(diagonalmatrix), x, α, β)
LinearAlgebra.mul!(x2, adjoint(M), x, α, β)
@test x1 ≈ x2
LinearAlgebra.mul!(x1, transpose(diagonalmatrix), x, α, β)
LinearAlgebra.mul!(x2, transpose(M), x, α, β)
@test x1 ≈ x2

diagonalmatrix[1, 1] = 1+im*2
@test diagonalmatrix[1, 1] == 1+im*2
diagonalmatrix[1, 3] = 1+im*2
@test diagonalmatrix[1, 3] == 1+im*2
diagonalmatrix[3, 1] = 2+im*1
@test diagonalmatrix[3, 1] == 2+im*1

# check threadsafecolors
block1 = randn(ComplexF64, 2, 2)
block2 = randn(ComplexF64, 2, 2)
block3 = randn(ComplexF64, 2, 2)

M = [
    zeros(2, 2) transpose(block1) transpose(block2)
    block1 zeros(2, 2) transpose(block2)
    block2 block3 zeros(2, 2)
]

od1 = BlockSparseMatrices.DenseMatrixBlock(block1, 3:4, 1:2)
od2 = BlockSparseMatrices.DenseMatrixBlock(block2, 5:6, 1:2)
od3 = BlockSparseMatrices.DenseMatrixBlock(block2, 5:6, 3:4)

diagonalmatrix = SymmetricBlockMatrix([], [od1, od2, od3], 6, 6)

@test !BlockSparseMatrices.issymthreadsafe()(od1, od2)
@test !BlockSparseMatrices.issymthreadsafe()(od2, od1)
@test !BlockSparseMatrices.issymthreadsafe()(od1, od3)
@test !BlockSparseMatrices.issymthreadsafe()(od3, od1)
@test BlockSparseMatrices.isthreadsafe()(od1, od3)
@test BlockSparseMatrices.isthreadsafe()(od3, od1)
@test !BlockSparseMatrices.issymthreadsafe()(od2, od3)
@test !BlockSparseMatrices.issymthreadsafe()(od3, od2)
