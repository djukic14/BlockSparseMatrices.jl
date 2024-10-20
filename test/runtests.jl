using BlockSparseMatrices
using Test

@testset "BlockSparseMatrices.jl" begin
    include("test_matrixblock.jl")
    include("test_blockmatrix.jl")
    include("test_symmetricblockmatrix.jl")
end
