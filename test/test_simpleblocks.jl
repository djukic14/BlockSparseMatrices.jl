using LinearAlgebra
using Test
using BlockSparseMatrices
using SparseArrays

block1 = randn(ComplexF64, 2, 2)
block2 = randn(ComplexF64, 2, 2)
block3 = randn(ComplexF64, 2, 2)
block4 = randn(ComplexF64, 2, 2)

M = [block1 block2; block3 block4]

denseblock1 = BlockSparseMatrices.DenseMatrixBlock(block1, 1:2, 1:2)
denseblock2 = BlockSparseMatrices.DenseMatrixBlock(block2, 1:2, 3:4)
denseblock3 = BlockSparseMatrices.DenseMatrixBlock(block3, 3:4, 1:2)
denseblock4 = BlockSparseMatrices.DenseMatrixBlock(block4, 3:4, 3:4)

adenseblock1 = adjoint(BlockSparseMatrices.DenseMatrixBlock(block1, 1:2, 1:2))
adenseblock2 = adjoint(BlockSparseMatrices.DenseMatrixBlock(block2, 1:2, 3:4))
adenseblock3 = adjoint(BlockSparseMatrices.DenseMatrixBlock(block3, 3:4, 1:2))
adenseblock4 = adjoint(BlockSparseMatrices.DenseMatrixBlock(block4, 3:4, 3:4))

tdenseblock1 = transpose(BlockSparseMatrices.DenseMatrixBlock(block1, 1:2, 1:2))
tdenseblock2 = transpose(BlockSparseMatrices.DenseMatrixBlock(block2, 1:2, 3:4))
tdenseblock3 = transpose(BlockSparseMatrices.DenseMatrixBlock(block3, 3:4, 1:2))
tdenseblock4 = transpose(BlockSparseMatrices.DenseMatrixBlock(block4, 3:4, 3:4))

@test eltype(denseblock1) == ComplexF64
@test eltype(typeof(denseblock1)) == ComplexF64
@test eltype(adenseblock1) == ComplexF64
@test eltype(typeof(adenseblock1)) == ComplexF64
@test eltype(tdenseblock1) == ComplexF64
@test eltype(typeof(tdenseblock1)) == ComplexF64

@test size(denseblock1) == (2, 2)
@test size(denseblock1, 1) == 2
@test size(denseblock1, 2) == 2
@test size(adenseblock1) == (2, 2)
@test size(adenseblock1, 1) == 2
@test size(adenseblock1, 2) == 2
@test size(tdenseblock1) == (2, 2)
@test size(tdenseblock1, 1) == 2
@test size(tdenseblock1, 2) == 2

@test rowindices(denseblock2) == 1:2
@test rowindices(adenseblock2) == 3:4
@test rowindices(tdenseblock2) == 3:4
@test colindices(denseblock2) == 3:4
@test colindices(adenseblock2) == 1:2
@test colindices(tdenseblock2) == 1:2
@test BlockSparseMatrices.matrix(denseblock1) == block1
@test BlockSparseMatrices.matrix(adenseblock1) == adjoint(block1)
@test BlockSparseMatrices.matrix(tdenseblock1) == transpose(block1)

@test nnz(denseblock1) == 4
@test nnz(adenseblock1) == 4
@test nnz(tdenseblock1) == 4

@test denseblock1 * [1, 0] == block1 * [1, 0]
@test denseblock1 * [0, 1] == block1 * [0, 1]

@test adenseblock1 * [1, 0] == adjoint(block1) * [1, 0]
@test adenseblock1 * [0, 1] == adjoint(block1) * [0, 1]

@test tdenseblock1 * [1, 0] == transpose(block1) * [1, 0]
@test tdenseblock1 * [0, 1] == transpose(block1) * [0, 1]

@test denseblock2 * [0, 1] == block2 * [0, 1]
@test denseblock2 * [1, 0] == block2 * [1, 0]

# @test adjoint(denseblock2) * [0, 1] == adjoint(block2) * [0, 1]
# @test adjoint(denseblock2) * [1, 0] == adjoint(block2) * [1, 0]

# @test transpose(denseblock2) * [0, 1] == transpose(block2) * [0, 1]
# @test transpose(denseblock2) * [1, 0] == transpose(block2) * [1, 0]

# @test (adenseblock2 * [0, 1])[3:4] == adjoint(block2) * [0, 1]
# @test (adenseblock2 * [0, 1])[1:2] == zeros(ComplexF64, 2)
# @test (adenseblock2 * [1, 0])[3:4] == adjoint(block2) * [1, 0]
# @test (adenseblock2 * [1, 0])[1:2] == zeros(ComplexF64, 2)

# @test (tdenseblock2 * [0, 1])[3:4] == transpose(block2) * [0, 1]
# @test (tdenseblock2 * [0, 1])[1:2] == zeros(ComplexF64, 2)
# @test (tdenseblock2 * [1, 0])[3:4] == transpose(block2) * [1, 0]
# @test (tdenseblock2 * [1, 0])[1:2] == zeros(ComplexF64, 2)

sparsematrix = BlockSparseMatrix([denseblock1, denseblock2, denseblock3, denseblock4], 4, 4)

@test eltype(sparsematrix) == ComplexF64
@test eltype(eltype(sparsematrix)) == ComplexF64
@test size(sparsematrix) == (4, 4)
@test size(sparsematrix, 1) == 4
@test size(sparsematrix, 2) == 4
@test nnz(sparsematrix) == 16

x = [1, 0, 0, 0]
for i in 1:4
    y = zeros(ComplexF64, 4)
    y[i] = 1
    @test sparsematrix * y == M * y
    @test adjoint(sparsematrix) * y == adjoint(M) * y
    @test transpose(sparsematrix) * y == transpose(M) * y

    x = randn(ComplexF64, 4)
    @test sparsematrix * x ≈ M * x
    @test adjoint(sparsematrix) * x ≈ adjoint(M) * x
    @test transpose(sparsematrix) * x ≈ transpose(M) * x

    x1 = randn(ComplexF64, 4)
    x2 = deepcopy(x1)
    x3 = deepcopy(x1)

    α = randn(ComplexF64)
    β = randn(ComplexF64)

    LinearAlgebra.mul!(x1, sparsematrix, x, α, β)
    LinearAlgebra.mul!(x2, M, x, α, β)

    LinearAlgebra.mul!(x3, adjoint(sparsematrix), x, α, β)
    LinearAlgebra.mul!(x3, adjoint(M), x, α, β)

    LinearAlgebra.mul!(x3, transpose(sparsematrix), x, α, β)
    LinearAlgebra.mul!(x3, transpose(M), x, α, β)

    @test x1 ≈ x2
end
