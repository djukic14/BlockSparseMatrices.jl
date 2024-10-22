using Test, TestItems, TestItemRunner
using BlockSparseMatrices
@testitem "BlockSparseMatrices" begin
    include("test_matrixblock.jl")
    include("test_blockmatrix.jl")
    include("test_symmetricblockmatrix.jl")
end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(BlockSparseMatrices; deps_compat=false)
end
@testitem "Code linting (JET.jl)" begin
    using JET
    JET.test_package(BlockSparseMatrices; target_defined_modules=true)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    pkgpath = pkgdir(BlockSparseMatrices)
    @test JuliaFormatter.format(pkgpath, overwrite=false)
end

@run_package_tests verbose = true
