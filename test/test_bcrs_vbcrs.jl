using Test
using BlockSparseMatrices
import BlockSparseMatrices: rowblockids, colblockids
using LinearAlgebra
using SparseArrays
using JLD2
using OhMyThreads

@testset "BlockCompressedRowStorage" begin
    @testset "Construction and basic properties" begin
        # Create simple 6×6 matrix with 2×2 blocks
        blocks = [
            [1.0 2.0; 3.0 4.0],
            [5.0 6.0; 7.0 8.0],
            [9.0 10.0; 11.0 12.0],
            [13.0 14.0; 15.0 16.0],
        ]
        rowptr = [1, 3, 5]  # Two block rows
        colindices = [1, 2, 1, 3]  # Block column positions
        blocksize = (2, 2)
        matsize = (4, 6)

        bcrs = BlockCompressedRowStorage(blocks, rowptr, colindices, blocksize, matsize)

        @test size(bcrs) == (4, 6)
        @test eltype(bcrs) == Float64
        @test nnz(bcrs) == 16
    end

    @testset "Element access" begin
        blocks = [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]]
        rowptr = [1, 3]
        colindices = [1, 2]
        blocksize = (2, 2)
        matsize = (2, 4)

        bcrs = BlockCompressedRowStorage(blocks, rowptr, colindices, blocksize, matsize)

        # Test element access
        @test bcrs[1, 1] == 1.0
        @test bcrs[1, 2] == 2.0
        @test bcrs[2, 1] == 3.0
        @test bcrs[2, 2] == 4.0
        @test bcrs[1, 3] == 5.0
        @test bcrs[1, 4] == 6.0
        @test bcrs[2, 3] == 7.0
        @test bcrs[2, 4] == 8.0
    end

    @testset "Matrix-vector multiplication" begin
        blocks = [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0], [9.0 10.0; 11.0 12.0]]
        rowptr = [1, 3, 4]
        colindices = [1, 2, 1]
        blocksize = (2, 2)
        matsize = (4, 4)

        bcrs = BlockCompressedRowStorage(blocks, rowptr, colindices, blocksize, matsize)

        x = ones(4)
        y = zeros(4)

        mul!(y, bcrs, x)

        # Verify result by constructing dense matrix
        dense = zeros(4, 4)
        dense[1:2, 1:2] = blocks[1]
        dense[1:2, 3:4] = blocks[2]
        dense[3:4, 1:2] = blocks[3]

        @test y ≈ dense * x

        # Test with α and β
        y2 = ones(4)
        mul!(y2, bcrs, x, 2.0, 0.5)
        @test y2 ≈ 2.0 * (dense * x) + 0.5 * ones(4)
    end

    @testset "Conversion from BlockSparseMatrix" begin
        # Create a BlockSparseMatrix
        blocks_vec = [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0], [9.0 10.0; 11.0 12.0]]
        rowindices = [[1, 2], [1, 2], [3, 4]]
        colindices = [[1, 2], [3, 4], [1, 2]]

        bsm = BlockSparseMatrix(blocks_vec, rowindices, colindices, (4, 4))

        # Convert to BCRS
        bcrs = BlockCompressedRowStorage(bsm, (2, 2))

        @test size(bcrs) == size(bsm)

        # Test matrix-vector product consistency
        x = rand(4)
        y_bsm = zeros(4)
        y_bcrs = zeros(4)

        mul!(y_bsm, bsm, x)
        mul!(y_bcrs, bcrs, x)

        @test y_bsm ≈ y_bcrs
    end

    @testset "Bounds checking" begin
        blocks = [[1.0 2.0; 3.0 4.0]]
        rowptr = [1, 2]
        colindices = [1]
        blocksize = (2, 2)
        matsize = (2, 2)

        bcrs = BlockCompressedRowStorage(blocks, rowptr, colindices, blocksize, matsize)

        @test_throws BoundsError bcrs[3, 1]
        @test_throws BoundsError bcrs[1, 3]
    end
end

@testset "VariableBlockCompressedRowStorage" begin
    @testset "Construction and basic properties" begin
        # Create matrix with variable-sized blocks
        blocks = [
            [1.0 2.0; 3.0 4.0],        # 2×2
            [5.0 6.0 7.0; 8.0 9.0 10.0],  # 2×3
            [11.0 12.0; 13.0 14.0; 15.0 16.0],  # 3×2
        ]
        rowptr = [1, 3, 4]
        colindices = [1, 3, 1]
        rowindices = [1, 1, 3]
        blocksizes = [(2, 2), (2, 3), (3, 2)]
        matsize = (5, 5)

        vbcrs = VariableBlockCompressedRowStorage(
            blocks, rowptr, colindices, rowindices, blocksizes, matsize
        )

        @test size(vbcrs) == (5, 5)
        @test eltype(vbcrs) == Float64
        @test nnz(vbcrs) == 2 * 2 + 2 * 3 + 3 * 2
    end

    @testset "Element access" begin
        blocks = [[1.0 2.0; 3.0 4.0], [5.0 6.0 7.0; 8.0 9.0 10.0]]
        rowptr = [1, 3]
        colindices = [1, 3]
        rowindices = [1, 1]
        blocksizes = [(2, 2), (2, 3)]
        matsize = (2, 5)

        vbcrs = VariableBlockCompressedRowStorage(
            blocks, rowptr, colindices, rowindices, blocksizes, matsize
        )

        # Test access in first block
        @test vbcrs[1, 1] == 1.0
        @test vbcrs[1, 2] == 2.0
        @test vbcrs[2, 1] == 3.0
        @test vbcrs[2, 2] == 4.0

        # Test access in second block
        @test vbcrs[1, 3] == 5.0
        @test vbcrs[1, 4] == 6.0
        @test vbcrs[1, 5] == 7.0
        @test vbcrs[2, 3] == 8.0
        @test vbcrs[2, 4] == 9.0
        @test vbcrs[2, 5] == 10.0
    end

    @testset "Matrix-vector multiplication" begin
        blocks = [[1.0 2.0; 3.0 4.0], [5.0 6.0 7.0], [8.0; 9.0; 10.0]]
        rowptr = [1, 3, 4]
        colindices = [1, 3, 1]
        rowindices = [1, 1, 3]
        blocksizes = [(2, 2), (1, 3), (3, 1)]
        matsize = (5, 5)

        vbcrs = VariableBlockCompressedRowStorage(
            blocks, rowptr, colindices, rowindices, blocksizes, matsize
        )

        x = ones(5)
        y = zeros(5)

        mul!(y, vbcrs, x)

        # Verify result by constructing dense matrix
        dense = zeros(5, 5)
        dense[1:2, 1:2] = blocks[1]
        dense[1:1, 3:5] = blocks[2]
        dense[3:5, 1:1] = blocks[3]

        @test y ≈ dense * x

        # Test with α and β
        y2 = ones(5)
        mul!(y2, vbcrs, x, 2.0, 0.5)
        @test y2 ≈ 2.0 * (dense * x) + 0.5 * ones(5)
    end

    @testset "Conversion from BlockSparseMatrix" begin
        # Create a BlockSparseMatrix with variable-sized blocks
        blocks_vec = [
            [1.0 2.0; 3.0 4.0],
            [5.0 6.0 7.0; 8.0 9.0 10.0],
            [11.0 12.0; 13.0 14.0; 15.0 16.0],
        ]
        rowindices = [[1, 2], [1, 2], [3, 4, 5]]
        colindices = [[1, 2], [3, 4, 5], [1, 2]]

        bsm = BlockSparseMatrix(blocks_vec, rowindices, colindices, (5, 5))

        # Convert to VBCRS
        vbcrs = VariableBlockCompressedRowStorage(bsm)

        @test size(vbcrs) == size(bsm)

        # Test matrix-vector product consistency
        x = rand(5)
        y_bsm = zeros(5)
        y_vbcrs = zeros(5)

        mul!(y_bsm, bsm, x)
        mul!(y_vbcrs, vbcrs, x)

        @test y_bsm ≈ y_vbcrs
    end

    @testset "Index dictionaries" begin
        blocks = [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]]
        rowptr = [1, 3]
        colindices = [1, 3]
        rowindices = [1, 1]
        blocksizes = [(2, 2), (2, 2)]
        matsize = (2, 4)

        vbcrs = VariableBlockCompressedRowStorage(
            blocks, rowptr, colindices, rowindices, blocksizes, matsize
        )

        # Test row index dictionary
        @test 1 in rowblockids(vbcrs, 1)
        @test 2 in rowblockids(vbcrs, 1)
        @test 1 in rowblockids(vbcrs, 2)
        @test 2 in rowblockids(vbcrs, 2)

        # Test column index dictionary
        @test 1 in colblockids(vbcrs, 1)
        @test 1 in colblockids(vbcrs, 2)
        @test 2 in colblockids(vbcrs, 3)
        @test 2 in colblockids(vbcrs, 4)
    end

    @testset "Bounds checking" begin
        blocks = [[1.0 2.0; 3.0 4.0]]
        rowptr = [1, 2]
        colindices = [1]
        rowindices = [1]
        blocksizes = [(2, 2)]
        matsize = (2, 2)

        vbcrs = VariableBlockCompressedRowStorage(
            blocks, rowptr, colindices, rowindices, blocksizes, matsize
        )

        @test_throws BoundsError vbcrs[3, 1]
        @test_throws BoundsError vbcrs[1, 3]
    end
end

@testset "BCRS vs VBCRS comparison" begin
    # Create the same matrix in both formats
    blocks_vec = [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0], [9.0 10.0; 11.0 12.0]]

    # BCRS format
    bcrs_blocks = [blocks_vec[1], blocks_vec[2], blocks_vec[3]]
    bcrs = BlockCompressedRowStorage(bcrs_blocks, [1, 3, 4], [1, 2, 1], (2, 2), (4, 4))

    # VBCRS format
    vbcrs = VariableBlockCompressedRowStorage(
        bcrs_blocks, [1, 3, 4], [1, 3, 1], [1, 1, 3], [(2, 2), (2, 2), (2, 2)], (4, 4)
    )

    # Test matrix-vector product consistency
    x = rand(4)
    y_bcrs = zeros(4)
    y_vbcrs = zeros(4)

    mul!(y_bcrs, bcrs, x)
    mul!(y_vbcrs, vbcrs, x)

    @test y_bcrs ≈ y_vbcrs
end

# Tests using blockexample.jld2 data
@testset "BlockCompressedRowStorage with blockexample.jld2" begin
    blockdict = load(
        joinpath(pkgdir(BlockSparseMatrices), "test", "assets", "blockexamples.jld2")
    )["blockdict"]

    for example in ["sphere", "cuboid"]
        (blocks, testindices, trialindices) = blockdict[example]

        # Determine block size (assuming uniform blocks)
        blocksize = (length(testindices[1]), length(trialindices[1]))

        # Verify all blocks have the same size
        all_same_size = all(
            (length(ti), length(tri)) == blocksize for
            (ti, tri) in zip(testindices, trialindices)
        )

        if !all_same_size
            @info "Skipping BCRS test for $example: non-uniform block sizes"
            continue
        end

        size1 = maximum(maximum, testindices)
        size2 = maximum(maximum, trialindices)

        # Create BlockSparseMatrix as reference
        b = BlockSparseMatrix(
            blocks, testindices, trialindices, (size1, size2); scheduler=SerialScheduler()
        )

        # Convert to BCRS
        bcrs = BlockCompressedRowStorage(b, blocksize)
        bcrs_parallel = BlockCompressedRowStorage(
            bcrs.blocks,
            bcrs.rowptr,
            bcrs.colindices,
            bcrs.blocksize,
            bcrs.size;
            scheduler=DynamicScheduler(),
        )

        @test_nowarn println(bcrs)
        @test_nowarn println(bcrs_parallel)

        bsparse = sparse(b)

        # Test basic properties
        @test size(bcrs) == size(b)
        @test size(bcrs_parallel) == size(b)
        @test nnz(bcrs) <= nnz(bsparse)
        @test nnz(bcrs_parallel) == nnz(bcrs)

        # Test matrix-vector multiplication
        for i in 1:10
            y = randn(ComplexF64, size2)
            x_bsm = zeros(ComplexF64, size1)
            x_bcrs = zeros(ComplexF64, size1)
            x_bcrs_parallel = zeros(ComplexF64, size1)

            @test b * y ≈ bcrs * y
            @test b * y ≈ bcrs_parallel * y

            # Test mul! with α and β
            @test LinearAlgebra.mul!(x_bsm, b, y, im, 2im) ≈
                LinearAlgebra.mul!(x_bcrs, bcrs, y, im, 2im)

            @test LinearAlgebra.mul!(x_bsm, b, y, im, 2im) ≈
                LinearAlgebra.mul!(x_bcrs_parallel, bcrs_parallel, y, im, 2im)
        end

        # Test element access
        rows = rowvals(bsparse)
        vals = nonzeros(bsparse)
        for j in 1:min(size(bsparse, 1), 100)  # Sample to avoid slowness
            for i in nzrange(bsparse, j)
                row = rows[i]
                @test vals[i] ≈ bcrs[row, j] atol = 1e-13
                @test vals[i] ≈ bcrs_parallel[row, j] atol = 1e-13
            end
        end
    end
end

@testset "VariableBlockCompressedRowStorage with blockexample.jld2" begin
    blockdict = load(
        joinpath(pkgdir(BlockSparseMatrices), "test", "assets", "blockexamples.jld2")
    )["blockdict"]

    for example in ["sphere", "cuboid"]
        (blocks, testindices, trialindices) = blockdict[example]

        size1 = maximum(maximum, testindices)
        size2 = maximum(maximum, trialindices)

        # Create BlockSparseMatrix as reference
        b = BlockSparseMatrix(
            blocks, testindices, trialindices, (size1, size2); scheduler=SerialScheduler()
        )

        # Convert to VBCRS
        vbcrs = VariableBlockCompressedRowStorage(b)
        vbcrs_parallel = VariableBlockCompressedRowStorage(
            vbcrs.blocks,
            vbcrs.rowptr,
            vbcrs.colindices,
            vbcrs.rowindices,
            vbcrs.blocksizes,
            vbcrs.size;
            scheduler=DynamicScheduler(),
        )

        @test_nowarn println(vbcrs)
        @test_nowarn println(vbcrs_parallel)

        bsparse = sparse(b)

        # Test basic properties
        @test size(vbcrs) == size(b)
        @test size(vbcrs_parallel) == size(b)
        @test nnz(vbcrs) <= nnz(bsparse)
        @test nnz(vbcrs_parallel) == nnz(vbcrs)

        # Test matrix-vector multiplication
        for i in 1:10
            y = randn(ComplexF64, size2)
            x_bsm = zeros(ComplexF64, size1)
            x_vbcrs = zeros(ComplexF64, size1)
            x_vbcrs_parallel = zeros(ComplexF64, size1)

            @test b * y ≈ vbcrs * y
            @test b * y ≈ vbcrs_parallel * y

            # Test mul! with α and β
            @test LinearAlgebra.mul!(x_bsm, b, y, im, 2im) ≈
                LinearAlgebra.mul!(x_vbcrs, vbcrs, y, im, 2im)

            @test LinearAlgebra.mul!(x_bsm, b, y, im, 2im) ≈
                LinearAlgebra.mul!(x_vbcrs_parallel, vbcrs_parallel, y, im, 2im)
        end

        # Test element access
        rows = rowvals(bsparse)
        vals = nonzeros(bsparse)
        for j in 1:min(size(bsparse, 1), 100)  # Sample to avoid slowness
            for i in nzrange(bsparse, j)
                row = rows[i]
                @test vals[i] ≈ vbcrs[row, j] atol = 1e-13
                @test vals[i] ≈ vbcrs_parallel[row, j] atol = 1e-13
            end
        end

        # Test index dictionaries
        for global_row in 1:min(size1, 50)  # Sample to avoid slowness
            blockids = rowblockids(vbcrs, global_row)
            @test !isempty(blockids) || all(iszero(bsparse[global_row, :]))
        end

        for global_col in 1:min(size2, 50)
            blockids = colblockids(vbcrs, global_col)
            @test !isempty(blockids) || all(iszero(bsparse[:, global_col]))
        end
    end
end

@testset "BCRS/VBCRS vs BlockSparseMatrix consistency" begin
    blockdict = load(
        joinpath(pkgdir(BlockSparseMatrices), "test", "assets", "blockexamples.jld2")
    )["blockdict"]

    for example in ["sphere", "cuboid"]
        (blocks, testindices, trialindices) = blockdict[example]

        size1 = maximum(maximum, testindices)
        size2 = maximum(maximum, trialindices)

        # Create all three formats
        bsm = BlockSparseMatrix(
            blocks, testindices, trialindices, (size1, size2); scheduler=SerialScheduler()
        )

        vbcrs = VariableBlockCompressedRowStorage(bsm)

        # Test that all formats produce the same result
        x = randn(ComplexF64, size2)
        y_bsm = bsm * x
        y_vbcrs = vbcrs * x

        @test y_bsm ≈ y_vbcrs atol = 1e-12

        # Check if blocks are uniform for BCRS test
        blocksize = (length(testindices[1]), length(trialindices[1]))
        all_same_size = all(
            (length(ti), length(tri)) == blocksize for
            (ti, tri) in zip(testindices, trialindices)
        )

        if all_same_size
            bcrs = BlockCompressedRowStorage(bsm, blocksize)
            y_bcrs = bcrs * x
            @test y_bsm ≈ y_bcrs atol = 1e-12
        end
    end
end
