using Test, TestItems, TestItemRunner

@testitem "BlockSparseMatrices" begin
    include("test_blockmatrix.jl")
    include("test_symmetricblockmatrix.jl")
end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(BlockSparseMatrices; unbound_args=false)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    using BlockSparseMatrices
    @test JuliaFormatter.format(pkgdir(BlockSparseMatrices), overwrite=false)
end

@run_package_tests verbose = true
