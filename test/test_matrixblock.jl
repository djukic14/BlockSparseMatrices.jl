using LinearAlgebra
using Test
using BlockSparseMatrices
using SparseArrays

block1 = randn(ComplexF64, 2, 3)

denseblock1 = BlockSparseMatrices.DenseMatrixBlock(block1, 1:2, 1:3)
adenseblock1 = adjoint(denseblock1)
tdenseblock1 = transpose(denseblock1)

@test eltype(denseblock1) == ComplexF64
@test eltype(typeof(denseblock1)) == ComplexF64
@test eltype(adenseblock1) == ComplexF64
@test eltype(typeof(adenseblock1)) == ComplexF64
@test eltype(tdenseblock1) == ComplexF64
@test eltype(typeof(tdenseblock1)) == ComplexF64

@test size(denseblock1) == (2, 3)
@test size(denseblock1, 1) == 2
@test size(denseblock1, 2) == 3
@test size(adenseblock1) == (3, 2)
@test size(adenseblock1, 1) == 3
@test size(adenseblock1, 2) == 2
@test size(tdenseblock1) == (3, 2)
@test size(tdenseblock1, 1) == 3
@test size(tdenseblock1, 2) == 2

@test rowindices(denseblock1) == 1:2
@test rowindices(adenseblock1) == 1:3
@test rowindices(tdenseblock1) == 1:3
@test colindices(denseblock1) == 1:3
@test colindices(adenseblock1) == 1:2
@test colindices(tdenseblock1) == 1:2
@test BlockSparseMatrices.matrix(denseblock1) == block1
@test BlockSparseMatrices.matrix(adenseblock1) == adjoint(block1)
@test BlockSparseMatrices.matrix(tdenseblock1) == transpose(block1)

@test nnz(denseblock1) == 6
@test nnz(adenseblock1) == 6
@test nnz(tdenseblock1) == 6

@test denseblock1 * [1, 0, 0] == block1 * [1, 0, 0]
@test denseblock1 * [0, 1, 0] == block1 * [0, 1, 0]
@test denseblock1 * [0, 0, 1] == block1 * [0, 0, 1]

@test adenseblock1 * [1, 0] == adjoint(block1) * [1, 0]
@test adenseblock1 * [0, 1] == adjoint(block1) * [0, 1]

@test tdenseblock1 * [1, 0] == transpose(block1) * [1, 0]
@test tdenseblock1 * [0, 1] == transpose(block1) * [0, 1]
