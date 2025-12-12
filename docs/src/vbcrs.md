# VariableBlockCompressedRowStorage

`VariableBlockCompressedRowStorage` is a memory‑efficient storage format for block‑sparse matrices where blocks are aligned along **contiguous row and column ranges**.  
This format is particularly useful when the matrix structure naturally groups into rectangular blocks with varying sizes.

!!! note
    The indices **must** be contiguous ranges for VBCRS, and **no sanity check is performed** during construction.

Below you will find a compact, step‑by‑step guide that shows how to

1. **Build** a VBCRS matrix from a list of blocks and their starting indices.  
2. **Multiply** it with vectors (including transposes).  
3. **Convert** it from a `BlockSparseMatrix`.
4. **Convert** it to a `SparseMatrixCSC`.

---

## Constructing a `VariableBlockCompressedRowStorage`

The key difference from `BlockSparseMatrix` is that VBCRS only needs the **first index** of each contiguous range rather than the full list of indices.

```@example vbcrs1
using CompScienceMeshes, BEAST, H2Trees #hide
using UnicodePlots #hide
using BlockSparseMatrices #hide

function sortbasis!(tree, space) #hide
    newindices = _sortbasisindices!(tree, space) #hide
    space.fns .= space.fns[newindices] #hide
    space.pos .= space.pos[newindices] #hide
    return space #hide
end #hide

function _sortbasisindices!(tree, basis) #hide
    newindices = Vector{Int}(undef, numfunctions(basis)) #hide

    lastindex = 0 #hide
    for node in H2Trees.leaves(tree) #hide
        points = H2Trees.values(tree, node) #hide

        newpositionids = (1:length(points)) .+ lastindex #hide

        newindices[points] .= newpositionids #hide

        tree(node).data.values .= newpositionids #hide

        lastindex += length(points) #hide
    end #hide

    return newindices #hide
end #hide

m = meshcuboid(1.0, 1.0, 1.0, 0.04) #hide
X = raviartthomas(m) #hide
tree = TwoNTree(X, 0.05) #hide
X = sortbasis!(tree, X) #hide

for leaf in H2Trees.leaves(tree) #hide
    vals = H2Trees.values(tree, leaf) #hide
    @assert vals == vals[begin]:vals[end] #hide
end #hide

testindices = Vector{Int}[] #hide
trialindices = Vector{Int}[] #hide

for node in H2Trees.leaves(tree) #hide
    for nearnode in H2Trees.NearNodeIterator(tree, node) #hide
        push!(testindices, collect(H2Trees.values(tree, node))) #hide
        push!(trialindices, collect(H2Trees.values(tree, nearnode))) #hide
    end #hide
end #hide

for i in eachindex(testindices) #hide
    sort!(testindices[i]) #hide
    sort!(trialindices[i]) #hide
    @assert testindices[i] == testindices[i][begin]:testindices[i][end] #hide
    @assert trialindices[i] == trialindices[i][begin]:trialindices[i][end] #hide
end #hide

blocks = Matrix{ComplexF64}[] #hide
for i in eachindex(testindices) #hide
    push!(blocks, randn(ComplexF64, length(testindices[i]), length(trialindices[i]))) #hide
end #hide
sizematrix = (numfunctions(X), numfunctions(X)) #hide

B = BlockSparseMatrix(blocks, testindices, trialindices, sizematrix)
```

Here, `testindices[1]` and `trialindices[1]` show contiguous ranges for the first block

```@example vbcrs2
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

function sortbasis!(tree, space) #hide
    newindices = _sortbasisindices!(tree, space) #hide
    space.fns .= space.fns[newindices] #hide
    space.pos .= space.pos[newindices] #hide
    return space #hide
end #hide

function _sortbasisindices!(tree, basis) #hide
    newindices = Vector{Int}(undef, numfunctions(basis)) #hide

    lastindex = 0 #hide
    for node in H2Trees.leaves(tree) #hide
        points = H2Trees.values(tree, node) #hide

        newpositionids = (1:length(points)) .+ lastindex #hide

        newindices[points] .= newpositionids #hide

        tree(node).data.values .= newpositionids #hide

        lastindex += length(points) #hide
    end #hide

    return newindices #hide
end #hide

m = meshcuboid(1.0, 1.0, 1.0, 0.04) #hide
X = raviartthomas(m) #hide
tree = TwoNTree(X, 0.05) #hide
X = sortbasis!(tree, X) #hide

for leaf in H2Trees.leaves(tree) #hide
    vals = H2Trees.values(tree, leaf) #hide
    @assert vals == vals[begin]:vals[end] #hide
end #hide

testindices = Vector{Int}[] #hide
trialindices = Vector{Int}[] #hide

for node in H2Trees.leaves(tree) #hide
    for nearnode in H2Trees.NearNodeIterator(tree, node) #hide
        push!(testindices, collect(H2Trees.values(tree, node))) #hide
        push!(trialindices, collect(H2Trees.values(tree, nearnode))) #hide
    end #hide
end #hide

for i in eachindex(testindices) #hide
    sort!(testindices[i]) #hide
    sort!(trialindices[i]) #hide
    @assert testindices[i] == testindices[i][begin]:testindices[i][end] #hide
    @assert trialindices[i] == trialindices[i][begin]:trialindices[i][end] #hide
end #hide

blocks = Matrix{ComplexF64}[] #hide
for i in eachindex(testindices) #hide
    push!(blocks, randn(ComplexF64, length(testindices[i]), length(trialindices[i]))) #hide
end #hide
sizematrix = (numfunctions(X), numfunctions(X)) #hide

testindices[1], trialindices[1] 
```

Now we can construct the VBCRS matrix by passing only the **first** index of each contiguous range

```@example vbcrs3
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

function sortbasis!(tree, space) #hide
    newindices = _sortbasisindices!(tree, space) #hide
    space.fns .= space.fns[newindices] #hide
    space.pos .= space.pos[newindices] #hide
    return space #hide
end #hide

function _sortbasisindices!(tree, basis) #hide
    newindices = Vector{Int}(undef, numfunctions(basis)) #hide

    lastindex = 0 #hide
    for node in H2Trees.leaves(tree) #hide
        points = H2Trees.values(tree, node) #hide

        newpositionids = (1:length(points)) .+ lastindex #hide

        newindices[points] .= newpositionids #hide

        tree(node).data.values .= newpositionids #hide

        lastindex += length(points) #hide
    end #hide

    return newindices #hide
end #hide

m = meshcuboid(1.0, 1.0, 1.0, 0.04) #hide
X = raviartthomas(m) #hide
tree = TwoNTree(X, 0.05) #hide
X = sortbasis!(tree, X) #hide

for leaf in H2Trees.leaves(tree) #hide
    vals = H2Trees.values(tree, leaf) #hide
    @assert vals == vals[begin]:vals[end] #hide
end #hide

testindices = Vector{Int}[] #hide
trialindices = Vector{Int}[] #hide

for node in H2Trees.leaves(tree) #hide
    for nearnode in H2Trees.NearNodeIterator(tree, node) #hide
        push!(testindices, collect(H2Trees.values(tree, node))) #hide
        push!(trialindices, collect(H2Trees.values(tree, nearnode))) #hide
    end #hide
end #hide

for i in eachindex(testindices) #hide
    sort!(testindices[i]) #hide
    sort!(trialindices[i]) #hide
    @assert testindices[i] == testindices[i][begin]:testindices[i][end] #hide
    @assert trialindices[i] == trialindices[i][begin]:trialindices[i][end] #hide
end #hide

blocks = Matrix{ComplexF64}[] #hide
for i in eachindex(testindices) #hide
    push!(blocks, randn(ComplexF64, length(testindices[i]), length(trialindices[i]))) #hide
end #hide
sizematrix = (numfunctions(X), numfunctions(X)) #hide

V = VariableBlockCompressedRowStorage(
    blocks, first.(testindices), first.(trialindices), sizematrix
)
```

*`V`* now behaves like a regular matrix but with significantly reduced storage overhead compared to storing full index arrays.

## Matrix–Vector Products

```@example vbcrs4
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

function sortbasis!(tree, space) #hide
    newindices = _sortbasisindices!(tree, space) #hide
    space.fns .= space.fns[newindices] #hide
    space.pos .= space.pos[newindices] #hide
    return space #hide
end #hide

function _sortbasisindices!(tree, basis) #hide
    newindices = Vector{Int}(undef, numfunctions(basis)) #hide

    lastindex = 0 #hide
    for node in H2Trees.leaves(tree) #hide
        points = H2Trees.values(tree, node) #hide

        newpositionids = (1:length(points)) .+ lastindex #hide

        newindices[points] .= newpositionids #hide

        tree(node).data.values .= newpositionids #hide

        lastindex += length(points) #hide
    end #hide

    return newindices #hide
end #hide

m = meshcuboid(1.0, 1.0, 1.0, 0.04) #hide
X = raviartthomas(m) #hide
tree = TwoNTree(X, 0.05) #hide
X = sortbasis!(tree, X) #hide

for leaf in H2Trees.leaves(tree) #hide
    vals = H2Trees.values(tree, leaf) #hide
    @assert vals == vals[begin]:vals[end] #hide
end #hide

testindices = Vector{Int}[] #hide
trialindices = Vector{Int}[] #hide

for node in H2Trees.leaves(tree) #hide
    for nearnode in H2Trees.NearNodeIterator(tree, node) #hide
        push!(testindices, collect(H2Trees.values(tree, node))) #hide
        push!(trialindices, collect(H2Trees.values(tree, nearnode))) #hide
    end #hide
end #hide

for i in eachindex(testindices) #hide
    sort!(testindices[i]) #hide
    sort!(trialindices[i]) #hide
    @assert testindices[i] == testindices[i][begin]:testindices[i][end] #hide
    @assert trialindices[i] == trialindices[i][begin]:trialindices[i][end] #hide
end #hide

blocks = Matrix{ComplexF64}[] #hide
for i in eachindex(testindices) #hide
    push!(blocks, randn(ComplexF64, length(testindices[i]), length(trialindices[i]))) #hide
end #hide
sizematrix = (numfunctions(X), numfunctions(X)) #hide

B = BlockSparseMatrix(blocks, testindices, trialindices, sizematrix) #hide

V = VariableBlockCompressedRowStorage( #hide
    blocks, first.(testindices), first.(trialindices), sizematrix #hide
) #hide

x = randn(ComplexF64, size(V,2)) #hide

@time V * x
@time V' * x
@time transpose(V) * x
nothing # hide
```

All three operations are implemented in pure Julia and respect the block‑sparsity, with **cache-friendly access patterns** due to the compressed row storage layout.

## Converting Between Formats

You can directly convert a `BlockSparseMatrix` to VBCRS format when the index structure satisfies the contiguity requirement

```@example vbcrs5
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide

function sortbasis!(tree, space) #hide
    newindices = _sortbasisindices!(tree, space) #hide
    space.fns .= space.fns[newindices] #hide
    space.pos .= space.pos[newindices] #hide
    return space #hide
end #hide

function _sortbasisindices!(tree, basis) #hide
    newindices = Vector{Int}(undef, numfunctions(basis)) #hide

    lastindex = 0 #hide
    for node in H2Trees.leaves(tree) #hide
        points = H2Trees.values(tree, node) #hide

        newpositionids = (1:length(points)) .+ lastindex #hide

        newindices[points] .= newpositionids #hide

        tree(node).data.values .= newpositionids #hide

        lastindex += length(points) #hide
    end #hide

    return newindices #hide
end #hide

m = meshcuboid(1.0, 1.0, 1.0, 0.04) #hide
X = raviartthomas(m) #hide
tree = TwoNTree(X, 0.05) #hide
X = sortbasis!(tree, X) #hide

for leaf in H2Trees.leaves(tree) #hide
    vals = H2Trees.values(tree, leaf) #hide
    @assert vals == vals[begin]:vals[end] #hide
end #hide

testindices = Vector{Int}[] #hide
trialindices = Vector{Int}[] #hide

for node in H2Trees.leaves(tree) #hide
    for nearnode in H2Trees.NearNodeIterator(tree, node) #hide
        push!(testindices, collect(H2Trees.values(tree, node))) #hide
        push!(trialindices, collect(H2Trees.values(tree, nearnode))) #hide
    end #hide
end #hide

for i in eachindex(testindices) #hide
    sort!(testindices[i]) #hide
    sort!(trialindices[i]) #hide
    @assert testindices[i] == testindices[i][begin]:testindices[i][end] #hide
    @assert trialindices[i] == trialindices[i][begin]:trialindices[i][end] #hide
end #hide

blocks = Matrix{ComplexF64}[] #hide
for i in eachindex(testindices) #hide
    push!(blocks, randn(ComplexF64, length(testindices[i]), length(trialindices[i]))) #hide
end #hide
sizematrix = (numfunctions(X), numfunctions(X)) #hide

B = BlockSparseMatrix(blocks, testindices, trialindices, sizematrix)

V = VariableBlockCompressedRowStorage(B) 
```

!!! note
    Conversion also works with `SymmetricBlockMatrix`.

Sometimes you need a `SparseMatrixCSC`. The conversion is straightforward

```@example vbcrs6
using CompScienceMeshes, BEAST, H2Trees # hide
using UnicodePlots # hide
using BlockSparseMatrices # hide
using SparseArrays # hide

function sortbasis!(tree, space) #hide
    newindices = _sortbasisindices!(tree, space) #hide
    space.fns .= space.fns[newindices] #hide
    space.pos .= space.pos[newindices] #hide
    return space #hide
end #hide

function _sortbasisindices!(tree, basis) #hide
    newindices = Vector{Int}(undef, numfunctions(basis)) #hide

    lastindex = 0 #hide
    for node in H2Trees.leaves(tree) #hide
        points = H2Trees.values(tree, node) #hide

        newpositionids = (1:length(points)) .+ lastindex #hide

        newindices[points] .= newpositionids #hide

        tree(node).data.values .= newpositionids #hide

        lastindex += length(points) #hide
    end #hide

    return newindices #hide
end #hide

m = meshcuboid(1.0, 1.0, 1.0, 0.04) #hide
X = raviartthomas(m) #hide
tree = TwoNTree(X, 0.05) #hide
X = sortbasis!(tree, X) #hide

for leaf in H2Trees.leaves(tree) #hide
    vals = H2Trees.values(tree, leaf) #hide
    @assert vals == vals[begin]:vals[end] #hide
end #hide

testindices = Vector{Int}[] #hide
trialindices = Vector{Int}[] #hide

for node in H2Trees.leaves(tree) #hide
    for nearnode in H2Trees.NearNodeIterator(tree, node) #hide
        push!(testindices, collect(H2Trees.values(tree, node))) #hide
        push!(trialindices, collect(H2Trees.values(tree, nearnode))) #hide
    end #hide
end #hide

for i in eachindex(testindices) #hide
    sort!(testindices[i]) #hide
    sort!(trialindices[i]) #hide
    @assert testindices[i] == testindices[i][begin]:testindices[i][end] #hide
    @assert trialindices[i] == trialindices[i][begin]:trialindices[i][end] #hide
end #hide

blocks = Matrix{ComplexF64}[] #hide
for i in eachindex(testindices) #hide
    push!(blocks, randn(ComplexF64, length(testindices[i]), length(trialindices[i]))) #hide
end #hide
sizematrix = (numfunctions(X), numfunctions(X)) #hide

B = BlockSparseMatrix(blocks, testindices, trialindices, sizematrix) #hide

V = VariableBlockCompressedRowStorage( 
    blocks, first.(testindices), first.(trialindices), sizematrix 
) 

s = sparse(V)
```

The VBCRS format is particularly memory‑efficient for this storage pattern, as it only stores the starting index of each contiguous block range rather than explicit lists of all indices.
