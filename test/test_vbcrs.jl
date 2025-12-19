using Test
using BlockSparseMatrices
using SparseArrays
using JLD2
using LinearAlgebra
using OhMyThreads
using UnicodePlots

@testset "VariableBlockCompressedRowStorage" begin
    blockdict = load(
        joinpath(pkgdir(BlockSparseMatrices), "test", "assets", "vbcrsexample.jld2")
    )["blockdict"]

    for example in ["sphere", "cuboid"]
        (blocks, testindices, trialindices) = blockdict[example]

        sze = (maximum(maximum, testindices), maximum(maximum, trialindices))

        b = BlockSparseMatrix(blocks, testindices, trialindices, sze;)

        for v in [
            VariableBlockCompressedRowStorage(
                blocks,
                first.(testindices),
                first.(trialindices),
                sze;
                scheduler=SerialScheduler(),
            ),
            VariableBlockCompressedRowStorage(b; scheduler=DynamicScheduler()),
        ]
            @test nnz(b) == nnz(v)

            for _ in 1:10
                x = randn(sze[2])
                @test maximum(abs, b * x - v * x) / maximum(abs, b * x) < 1e-13
                @test er = maximum(abs, b' * x - v' * x) / maximum(abs, b * x) < 1e-13
                @test er =
                    maximum(abs, transpose(b) * x - transpose(v) * x) /
                    maximum(abs, b * x) < 1e-13

                s = sparse(v)
                x = randn(sze[2])
                @test er = maximum(abs, s * x - v * x) / maximum(abs, s * x) < 1e-13
                @test er = maximum(abs, s' * x - v' * x) / maximum(abs, s * x) < 1e-13
                @test er =
                    maximum(abs, transpose(s) * x - transpose(v) * x) /
                    maximum(abs, s * x) < 1e-13
            end
        end
    end
end

@testset "SymmetricBlockMatrix to VariableBlockCompressedRowStorage" begin
    blockdict = load(
        joinpath(
            pkgdir(BlockSparseMatrices), "test", "assets", "symmetricvbcrsexample.jld2"
        ),
    )["blockdict"]

    for example in ["sphere", "cuboid"]
        (diagonalblocks, diagonalindices, offblocks, testindices, trialindices) = blockdict[example]

        sze = (maximum(maximum, diagonalindices), maximum(maximum, diagonalindices))

        s = SymmetricBlockMatrix(
            diagonalblocks, diagonalindices, offblocks, testindices, trialindices, sze
        )

        v = VariableBlockCompressedRowStorage(s)

        @test nnz(s) == nnz(v)

        for _ in 1:10
            x = randn(sze[2])
            @test maximum(abs, s * x - v * x) / maximum(abs, s * x) < 1e-13
            @test er = maximum(abs, s' * x - v' * x) / maximum(abs, s * x) < 1e-13
            @test er =
                maximum(abs, transpose(s) * x - transpose(v) * x) / maximum(abs, s * x) <
                1e-13

            s = sparse(v)
            x = randn(sze[2])
            @test er = maximum(abs, s * x - v * x) / maximum(abs, s * x) < 1e-13
            @test er = maximum(abs, s' * x - v' * x) / maximum(abs, s * x) < 1e-13
            @test er =
                maximum(abs, transpose(s) * x - transpose(v) * x) / maximum(abs, s * x) <
                1e-13
        end
    end
end
