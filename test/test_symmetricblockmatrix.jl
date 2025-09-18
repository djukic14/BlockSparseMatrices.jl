using Test
using BlockSparseMatrices
using SparseArrays
using JLD2
using LinearAlgebra
using OhMyThreads
using UnicodePlots
@testset "SymmetricBlockMatrix" begin
    blockdict = load(
        joinpath(
            pkgdir(BlockSparseMatrices), "test", "assets", "symmetricblockexamples.jld2"
        ),
    )["blockdict"]

    for example in ["sphere", "cuboid"]
        (diagonalblocks, selfindices, offblocks, testindices, trialindices) = blockdict[example]

        size1 = maximum(maximum, testindices)
        size2 = maximum(maximum, trialindices)
        b = SymmetricBlockMatrix(
            diagonalblocks,
            selfindices,
            offblocks,
            testindices,
            trialindices,
            (size1, size2);
            scheduler=SerialScheduler(),
        )

        @test_nowarn println(b)
        @test_nowarn println(b')
        @test_nowarn println(transpose(b))

        bparallel = SymmetricBlockMatrix(
            diagonalblocks,
            selfindices,
            offblocks,
            testindices,
            trialindices,
            (size1, size2);
            scheduler=DynamicScheduler(),
        )

        @test_nowarn println(bparallel)
        @test_nowarn println(bparallel')
        @test_nowarn println(transpose(bparallel))

        bsparse = sparse(b)
        @test issymmetric(bsparse)

        bsparsetranspose = permutedims(bsparse)
        bsparseadjoint = conj.(bsparsetranspose)

        @test maximum(abs, sparse(b[:, :]).nzval - bsparse.nzval) < 1e-13
        @test maximum(abs, sparse(bparallel[:, :]).nzval - bsparse.nzval) < 1e-13

        @test maximum(abs, sparse(adjoint(b)[:, :]) - bsparseadjoint) < 1e-13
        @test maximum(abs, sparse(adjoint(bparallel)[:, :]).nzval - bsparseadjoint.nzval) <
            1e-13

        @test maximum(abs, sparse(transpose(b)[:, :]).nzval - bsparsetranspose.nzval) <
            1e-13
        @test maximum(
            abs, sparse(transpose(bparallel)[:, :]).nzval - bsparsetranspose.nzval
        ) < 1e-13

        for i in 1:10
            y = randn(ComplexF64, size2)

            x = randn(ComplexF64, size1)

            @test b * y ≈ bsparse * y
            @test bparallel * y ≈ bsparse * y

            @test b' * y ≈ bsparse' * y
            @test bparallel' * y ≈ bsparse' * y

            @test transpose(b) * y ≈ transpose(bsparse) * y
            @test transpose(bparallel) * y ≈ transpose(bsparse) * y

            @test LinearAlgebra.mul!(x, b, y, im, 2im) ≈
                LinearAlgebra.mul!(x, bsparse, y, im, 2im)

            @test LinearAlgebra.mul!(x, bparallel, y, im, 2im) ≈
                LinearAlgebra.mul!(x, bsparse, y, im, 2im)

            @test LinearAlgebra.mul!(x, b', y, im, 2im) ≈
                LinearAlgebra.mul!(x, bsparse', y, im, 2im)

            @test LinearAlgebra.mul!(x, bparallel', y, im, 2im) ≈
                LinearAlgebra.mul!(x, bsparse', y, im, 2im)

            @test LinearAlgebra.mul!(x, transpose(b), y, im, 2im) ≈
                LinearAlgebra.mul!(x, transpose(bsparse), y, im, 2im)

            @test LinearAlgebra.mul!(x, transpose(bparallel), y, im, 2im) ≈
                LinearAlgebra.mul!(x, transpose(bsparse), y, im, 2im)
        end

        @test nnz(b) == nnz(bsparse)
        @test nnz(bparallel) == nnz(bsparse)

        @test nnz(b') == nnz(bsparse)
        @test nnz(bparallel') == nnz(bsparse)

        @test nnz(transpose(b)) == nnz(bsparse)
        @test nnz(transpose(bparallel)) == nnz(bsparse)

        rows = rowvals(bsparse)
        vals = nonzeros(bsparse)
        for j in 1:size(bsparse, 1)
            for i in nzrange(bsparse, j)
                row = rows[i]
                @test vals[i] == b[row, j]

                @test vals[i] == bparallel[row, j]
            end
        end

        bsparse.nzval .= 1
        negativebsparse = sparse(ones(size(bsparse)) - bsparse)

        counter = 0
        for j in 1:size(bsparse, 1)
            for i in nzrange(negativebsparse, j)
                row = rowvals(negativebsparse)[i]
                @test_throws ErrorException b[row, j] = 10
                @test_throws ErrorException bparallel[row, j] = 10

                @test iszero(b[row, j])
                @test iszero(bparallel[row, j])

                counter > 10 && break
                counter += 1
            end
        end

        for j in 1:size(bsparse, 1)
            for i in nzrange(bsparse, j)
                row = rows[i]
                b[row, j] = 0
                bparallel[row, j] = 0
            end
        end

        bcollected = b[:, :]
        @test all(iszero, bcollected)
        bparallelcollected = bparallel[:, :]
        @test all(iszero, bparallelcollected)
    end
end
