# SymmetricBlockMatrix

A `SymmetricBlockMatrix` is a matrices that is symmetric and (at least) **sparse at the block level**.  
This can, for example, occur when the matrix originates from a fast multipole method (FMM) or any other algorithm that naturally groups rows and columns into interacting blocks (e.g. near‑field interactions).

!!! note
    No sanity check is performed that makes sure that not both blocks of a pair of symmetric blocks are stored.

Below you will find a compact, step‑by‑step guide that shows how to

1. **Build** a symmetric block‑sparse matrix from two lists of dense blocks and their index maps.  
2. **Multiply** it with vectors (including transposes).  
3. **Convert** it to a standard `SparseMatrixCSC` for comparison.  
4. **Enable** multi‑threaded construction with `OhMyThreads`.

---

## Constructing a `SymmetricBlockMatrix`

```@example sparse1
using CompScienceMeshes, BEAST, H2Trees
using UnicodePlots
using BlockSparseMatrices

# make sure that only one of the two symmetric  blocks is stored
struct GalerkinSymmetricIsNearFunctor{N}
    isnear::N
end

function (f::GalerkinSymmetricIsNearFunctor)(tree, nodea, nodeb)
    if H2Trees.isleaf(tree, nodeb)
        return f.isnear(tree, nodea, nodeb) && nodea > nodeb
    else
        return f.isnear(tree, nodea, nodeb)
    end
end

m = meshsphere(1.0, 0.1)
X = raviartthomas(m)
sizeS = (numfunctions(X), numfunctions(X))
tree = TwoNTree(X, 0.2)

selfvalues, colvalues, rowvalues = H2Trees.nearinteractions(
    tree; extractselfvalues=true, isnear=GalerkinSymmetricIsNearFunctor(H2Trees.isnear)
)

diagonals = Matrix{Float64}[]
for i in eachindex(selfvalues)
    push!(diagonals, rand(Float64, length(selfvalues[i]), length(selfvalues[i])))
    diagonals[end] = (diagonals[end] + transpose(diagonals[end])) / 2  # diagonals are symmetric
end

offdiagonals = Matrix{Float64}[]
for i in eachindex(colvalues)
    push!(offdiagonals, rand(Float64, length(colvalues[i]), length(rowvalues[i])))
end

S = SymmetricBlockMatrix(diagonals, selfvalues, offdiagonals, colvalues, rowvalues, sizeS)
```

*`S`* now behaves like a regular matrix: you can query its size, inspect its blocks, etc.

## Matrix–Vector Products

```@example sparse2
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

# make sure that only one of the two symmetric  blocks is stored # hide
struct GalerkinSymmetricIsNearFunctor{N} # hide
    isnear::N # hide
end # hide

function (f::GalerkinSymmetricIsNearFunctor)(tree, nodea, nodeb) # hide
    if H2Trees.isleaf(tree, nodeb) # hide
        return f.isnear(tree, nodea, nodeb) && nodea > nodeb # hide
    else # hide
        return f.isnear(tree, nodea, nodeb) # hide
    end # hide
end # hide

m = meshsphere(1.0, 0.1) # hide
X = raviartthomas(m) # hide
sizeS = (numfunctions(X), numfunctions(X)) # hide
tree = TwoNTree(X, 0.2) # hide

selfvalues, colvalues, rowvalues = H2Trees.nearinteractions( # hide
    tree; extractselfvalues=true, isnear=GalerkinSymmetricIsNearFunctor(H2Trees.isnear) # hide
) # hide

diagonals = Matrix{Float64}[] # hide
for i in eachindex(selfvalues) # hide
    push!(diagonals, rand(Float64, length(selfvalues[i]), length(selfvalues[i]))) # hide
    diagonals[end] = (diagonals[end] + transpose(diagonals[end])) / 2  # diagonals are symmetric # hide
end # hide

offdiagonals = Matrix{Float64}[] # hide
for i in eachindex(colvalues) # hide
    push!(offdiagonals, rand(Float64, length(colvalues[i]), length(rowvalues[i]))) # hide
end # hide

S = SymmetricBlockMatrix(diagonals, selfvalues, offdiagonals, colvalues, rowvalues, sizeS) # hide

y = randn(Float64, numfunctions(X))
@time S * y
@time S' * y
@time transpose(S) * y
@show maximum(abs, S * y - transpose(S) * y)
nothing # hide
```

All three operations are implemented in pure Julia and respect the block‑sparsity, so they are typically **much faster** than converting the matrix to a dense format first.

## Converting to a Classical Sparse Matrix

Sometimes you need a `SparseMatrixCSC`. The conversion is straightforward. You can compare the memory footprints:

```@example sparse3
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

# make sure that only one of the two symmetric  blocks is stored # hide
struct GalerkinSymmetricIsNearFunctor{N} # hide
    isnear::N # hide
end # hide

function (f::GalerkinSymmetricIsNearFunctor)(tree, nodea, nodeb) # hide
    if H2Trees.isleaf(tree, nodeb) # hide
        return f.isnear(tree, nodea, nodeb) && nodea > nodeb # hide
    else # hide
        return f.isnear(tree, nodea, nodeb) # hide
    end # hide
end # hide

m = meshsphere(1.0, 0.1) # hide
X = raviartthomas(m) # hide
sizeS = (numfunctions(X), numfunctions(X)) # hide
tree = TwoNTree(X, 0.2) # hide

selfvalues, colvalues, rowvalues = H2Trees.nearinteractions( # hide
    tree; extractselfvalues=true, isnear=GalerkinSymmetricIsNearFunctor(H2Trees.isnear) # hide
) # hide

diagonals = Matrix{Float64}[] # hide
for i in eachindex(selfvalues) # hide
    push!(diagonals, rand(Float64, length(selfvalues[i]), length(selfvalues[i]))) # hide
    diagonals[end] = (diagonals[end] + transpose(diagonals[end])) / 2  # diagonals are symmetric # hide
end # hide

offdiagonals = Matrix{Float64}[] # hide
for i in eachindex(colvalues) # hide
    push!(offdiagonals, rand(Float64, length(colvalues[i]), length(rowvalues[i]))) # hide
end # hide

S = SymmetricBlockMatrix(diagonals, selfvalues, offdiagonals, colvalues, rowvalues, sizeS) # hide

using SparseArrays
Ssparse = sparse(S)
display(Ssparse);

@show Base.format_bytes(Base.summarysize(S))
@show Base.format_bytes(Base.summarysize(Ssparse))
nothing # hide
```

In specific examples `SymmetricBlockMatrix` consumes less memory.

## Multi‑Threaded Construction (Optional)

Depending on the example, the block-coloring step can be a bottleneck, thus multithreading is switched-off by default.
To enable multithreading you can determine a scheduler from [OhMyThreads](https://github.com/JuliaFolds2/OhMyThreads.jl).

```@example sparse4
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

# make sure that only one of the two symmetric  blocks is stored  # hide
struct GalerkinSymmetricIsNearFunctor{N} # hide
    isnear::N # hide
end # hide

function (f::GalerkinSymmetricIsNearFunctor)(tree, nodea, nodeb) # hide
    if H2Trees.isleaf(tree, nodeb) # hide
        return f.isnear(tree, nodea, nodeb) && nodea > nodeb # hide
    else # hide
        return f.isnear(tree, nodea, nodeb) # hide
    end # hide
end # hide

m = meshsphere(1.0, 0.1) # hide
X = raviartthomas(m) # hide
sizeS = (numfunctions(X), numfunctions(X)) # hide
tree = TwoNTree(X, 0.2) # hide

selfvalues, colvalues, rowvalues = H2Trees.nearinteractions(  # hide
    tree; extractselfvalues=true, isnear=GalerkinSymmetricIsNearFunctor(H2Trees.isnear)  # hide
) # hide

diagonals = Matrix{Float64}[] # hide
for i in eachindex(selfvalues) # hide
    push!(diagonals, rand(Float64, length(selfvalues[i]), length(selfvalues[i]))) # hide
    diagonals[end] = (diagonals[end] + transpose(diagonals[end])) / 2  # diagonals are symmetric  # hide
end  # hide

offdiagonals = Matrix{Float64}[] # hide
for i in eachindex(colvalues) # hide
    push!(offdiagonals, rand(Float64, length(colvalues[i]), length(rowvalues[i]))) # hide
end # hide

using OhMyThreads
S = SymmetricBlockMatrix(
    diagonals,
    selfvalues,
    offdiagonals,
    colvalues,
    rowvalues,
    sizeS;
    scheduler=DynamicScheduler(),
)
nothing # hide
```
