using Test, TestItems, TestItemRunner

@testitem "BlockSparseMatrices" begin
    include("test_matrixblock.jl")
    include("test_blockmatrix.jl")
    include("test_symmetricblockmatrix.jl")
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    pkgpath = pkgdir(BlockSparseMatrices)
    @test JuliaFormatter.format(pkgpath, overwrite=false)
end

@run_package_tests verbose = true
