# BlockSparseMatrix

A `BlockSparseMatrix` is a matrices that is (at least) **sparse at the block level**.  
This can, for example, occur when the matrix originates from a fast multipole method (FMM) or any other algorithm that naturally groups rows and columns into interacting blocks (e.g. near‑field interactions).

Below you will find a compact, step‑by‑step guide that shows how to

1. **Build** a block‑sparse matrix from a list of dense blocks and their index maps.  
2. **Multiply** it with vectors (including transposes).  
3. **Convert** it to a standard `SparseMatrixCSC` for comparison.  
4. **Enable** multi‑threaded construction with `OhMyThreads`.

---

## Constructing a `BlockSparseMatrix`

```@example block1
using CompScienceMeshes, BEAST, H2Trees
using UnicodePlots
using BlockSparseMatrices

m = meshsphere(1.0, 0.1)
X = raviartthomas(m)
tree = TwoNTree(X, 0.2)
colvalues, rowvalues = H2Trees.nearinteractions(tree)

blocks = Matrix{Float64}[]
for i in eachindex(colvalues)
    push!(blocks, randn(Float64, length(colvalues[i]), length(rowvalues[i])))
end

B = BlockSparseMatrix(blocks, colvalues, rowvalues, (numfunctions(X), numfunctions(X)))
```

*`B`* now behaves like a regular matrix: you can query its size, inspect its blocks, etc.

## Matrix–Vector Products

```@example block2
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

m = meshsphere(1.0, 0.1) # hide
X = raviartthomas(m) # hide
tree = TwoNTree(X, 0.2) # hide
colvalues, rowvalues = H2Trees.nearinteractions(tree) # hide

blocks = Matrix{Float64}[] # hide
for i in eachindex(colvalues) # hide
    push!(blocks, randn(Float64, length(colvalues[i]), length(rowvalues[i]))) # hide
end # hide

B = BlockSparseMatrix(blocks, colvalues, rowvalues, (numfunctions(X), numfunctions(X))) # hide

y = randn(Float64, size(B, 1))
@time B * y
@time B' * y
@time transpose(B) * y
nothing # hide
```

All three operations are implemented in pure Julia and respect the block‑sparsity, so they are typically **much faster** than converting the matrix to a dense format first.

## Converting to a Classical Sparse Matrix

Sometimes you need a `SparseMatrixCSC`. The conversion is straightforward. You can compare the memory footprints:

```@example block3
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

m = meshsphere(1.0, 0.1) # hide
X = raviartthomas(m) # hide
tree = TwoNTree(X, 0.2) # hide
colvalues, rowvalues = H2Trees.nearinteractions(tree) # hide

blocks = Matrix{Float64}[] # hide
for i in eachindex(colvalues) # hide
    push!(blocks, randn(Float64, length(colvalues[i]), length(rowvalues[i]))) # hide
end # hide

B = BlockSparseMatrix(blocks, colvalues, rowvalues, (numfunctions(X), numfunctions(X))) # hide

using SparseArrays
Bsparse = sparse(B)

@show Base.format_bytes(Base.summarysize(B));
@show Base.format_bytes(Base.summarysize(Bsparse));
nothing # hide
```

In specific examples `BlockSparseMatrix` consumes less memory.

## Multi‑Threaded Construction (Optional)

Depending on the example, the block-coloring step can be a bottleneck, thus multithreading is switched-off by default.
To enable multithreading you can determine a scheduler from [OhMyThreads](https://github.com/JuliaFolds2/OhMyThreads.jl).

```@example block3
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

m = meshsphere(1.0, 0.1) # hide
X = raviartthomas(m) # hide
tree = TwoNTree(X, 0.2) # hide
colvalues, rowvalues = H2Trees.nearinteractions(tree) # hide

blocks = Matrix{Float64}[] # hide
for i in eachindex(colvalues) # hide
    push!(blocks, randn(Float64, length(colvalues[i]), length(rowvalues[i]))) # hide
end # hide

using OhMyThreads
B = BlockSparseMatrix(
    blocks,
    colvalues,
    rowvalues,
    (numfunctions(X), numfunctions(X));
    scheduler=DynamicScheduler(),
)

nothing # hide
```
