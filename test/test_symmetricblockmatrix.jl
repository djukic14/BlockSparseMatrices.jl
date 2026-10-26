using Test
using BlockSparseMatrices
using SparseArrays
using JLD2
using LinearAlgebra
using OhMyThreads
using UnicodePlots
using GraphsColoring: WorkstreamGreedy
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
    end
end

@testset "SymmetricBlockMatrix constructors" begin
    for scheduler in (SerialScheduler(), DynamicScheduler())
        empty_matrix = SymmetricBlockMatrix{Float64}(
            [], [], [], [], [], (3, 3); scheduler=scheduler
        )
        @test size(empty_matrix) == (3, 3)
        @test eltype(empty_matrix) == Float64
        @test nnz(empty_matrix) == 0
        @test empty_matrix * ones(3) == zeros(3)
        @test transpose(empty_matrix) * ones(3) == zeros(3)
        @test adjoint(empty_matrix) * ones(3) == zeros(3)

        diagonal_matrix = SymmetricBlockMatrix(
            [Float64[2 0; 0 3]],
            [[1, 2]],
            Matrix{Float64}[],
            Vector{Int}[],
            Vector{Int}[],
            (2, 2);
            scheduler=scheduler,
        )
        @test diagonal_matrix * ones(2) ≈ [2.0, 3.0]
        @test transpose(diagonal_matrix) * ones(2) ≈ [2.0, 3.0]
        @test adjoint(diagonal_matrix) * ones(2) ≈ [2.0, 3.0]

        offdiagonal_matrix = SymmetricBlockMatrix(
            Matrix{Float64}[],
            Vector{Int}[],
            [reshape([4.0], 1, 1)],
            [[1]],
            [[2]],
            (2, 2);
            scheduler=scheduler,
        )
        @test offdiagonal_matrix * [1.0, 2.0] ≈ [8.0, 4.0]
        @test transpose(offdiagonal_matrix) * [1.0, 2.0] ≈ [8.0, 4.0]
        @test adjoint(offdiagonal_matrix) * [1.0, 2.0] ≈ [8.0, 4.0]
    end

    diagonal_indices = [[1]]
    row_indices = [[1]]
    col_indices = [[2]]
    serial_matrix = SymmetricBlockMatrix(
        [reshape([1.0], 1, 1)],
        diagonal_indices,
        [reshape([2.0], 1, 1)],
        row_indices,
        col_indices,
        (2, 2);
        scheduler=SerialScheduler(),
    )
    @test serial_matrix.diagonalcolors == [[1]]
    @test serial_matrix.offdiagonalcolors == [[1]]
    @test serial_matrix.transposeoffdiagonalcolors == [[1]]
    @test serial_matrix.diagonalindices === diagonal_indices
    @test serial_matrix.rowindices === row_indices
    @test serial_matrix.colindices === col_indices

    @test_throws ArgumentError SymmetricBlockMatrix([], [], [], [], [], (3, 3))

    @test_throws ArgumentError SymmetricBlockMatrix{Float64}(
        [reshape([1.0], 1, 1)],
        Vector{Int}[],
        Matrix{Float64}[],
        Vector{Int}[],
        Vector{Int}[],
        (2, 2);
        scheduler=SerialScheduler(),
    )
    @test_throws ArgumentError SymmetricBlockMatrix{Float64}(
        Matrix{Float64}[],
        Vector{Int}[],
        [reshape([2.0], 1, 1)],
        Vector{Int}[],
        [[2]],
        (2, 2);
        scheduler=SerialScheduler(),
    )
    @test_throws ArgumentError SymmetricBlockMatrix{Float64}(
        Matrix{Float64}[],
        Vector{Int}[],
        [reshape([2.0], 1, 1)],
        [[1]],
        Vector{Int}[],
        (2, 2);
        scheduler=SerialScheduler(),
    )

    greedy_matrix = SymmetricBlockMatrix(
        [reshape([1.0], 1, 1)],
        [[1]],
        [reshape([2.0], 1, 1)],
        [[1]],
        [[2]],
        (2, 2);
        scheduler=DynamicScheduler(),
        coloringalgorithm=WorkstreamGreedy,
    )
    @test greedy_matrix * [1.0, 2.0] ≈ [5.0, 2.0]
end
