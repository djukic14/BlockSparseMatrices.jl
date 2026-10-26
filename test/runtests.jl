using Test, TestItems, TestItemRunner

@testitem "BlockSparseMatrices" begin
    include("test_blockmatrix.jl")
    include("test_symmetricblockmatrix.jl")
    include("test_vbcrs.jl")
end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    # The Downgrade workflow promotes the test dependencies into [deps], where they would
    # otherwise be reported as stale because `using BlockSparseMatrices` does not load them.
    Aqua.test_all(
        BlockSparseMatrices;
        unbound_args=false,
        stale_deps=(;
            ignore=[
                :Aqua, :JLD2, :JuliaFormatter, :TestItemRunner, :TestItems, :UnicodePlots
            ]
        ),
    )
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter, BlockSparseMatrices
    @test JuliaFormatter.format(pkgdir(BlockSparseMatrices), overwrite=false)
end

@run_package_tests verbose = true
