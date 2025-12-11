using Test
using BlockSparseMatrices
using BEAST, CompScienceMeshes
using H2Trees
using SparseArrays
using OhMyThreads

function sortbasis!(tree, space)
    newindices = _sortbasisindices!(tree, space)
    space.fns .= space.fns[newindices]
    space.pos .= space.pos[newindices]
    return space
end

function _sortbasisindices!(tree, basis)
    newindices = Vector{Int}(undef, numfunctions(basis))

    lastindex = 0
    for node in H2Trees.leaves(tree)
        points = H2Trees.values(tree, node)

        newpositionids = (1:length(points)) .+ lastindex

        newindices[points] .= newpositionids

        tree(node).data.values .= newpositionids

        lastindex += length(points)
    end

    return newindices
end

m = meshsphere(1.0, 0.1)
X = raviartthomas(m)
tree = TwoNTree(X, 0.15)
X = sortbasis!(tree, X)
# tree = TwoNTree(X, 0.15)

for leaf in H2Trees.leaves(tree)
    vals = H2Trees.values(tree, leaf)
    @test vals == vals[begin]:vals[end]
end

testindices = Vector{Int}[]
trialindices = Vector{Int}[]

for node in H2Trees.leaves(tree)
    for nearnode in H2Trees.NearNodeIterator(tree, node)
        push!(testindices, collect(H2Trees.values(tree, node)))
        push!(trialindices, collect(H2Trees.values(tree, nearnode)))
    end
end

for i in eachindex(testindices)
    sort!(testindices[i])
    sort!(trialindices[i])
    @test testindices[i] == testindices[i][begin]:testindices[i][end]
    @test trialindices[i] == trialindices[i][begin]:trialindices[i][end]
end

blocks = Matrix{ComplexF64}[]
for i in eachindex(testindices)
    push!(blocks, randn(ComplexF64, length(testindices[i]), length(trialindices[i])))
end

b = BlockSparseMatrix(
    blocks,
    testindices,
    trialindices,
    (numfunctions(X), numfunctions(X));
    # scheduler=SerialScheduler(),
    # scheduler=DynamicScheduler(),
)

# v = VariableBlockCompressedRowStorage(b; scheduler=SerialScheduler())
v = VariableBlockCompressedRowStorage(
    blocks,
    first.(testindices),
    first.(trialindices),
    (numfunctions(X), numfunctions(X));
    # scheduler=SerialScheduler(),
    # scheduler=DynamicScheduler(),
)
@test nnz(b) == nnz(v)

x = randn(numfunctions(X))
@show er = maximum(abs, b * x - v * x) / maximum(abs, b * x);
@show er = maximum(abs, b' * x - v' * x) / maximum(abs, b * x);
@show er = maximum(abs, transpose(b) * x - transpose(v) * x) / maximum(abs, b * x);

s = sparse(v)
x = randn(numfunctions(X))
@show er = maximum(abs, s * x - v * x) / maximum(abs, s * x);
@show er = maximum(abs, s' * x - v' * x) / maximum(abs, s * x);
@show er = maximum(abs, transpose(s) * x - transpose(v) * x) / maximum(abs, s * x);

# using LinearAlgebra
# y = zeros(ComplexF64, numfunctions(X))
# b1 = @benchmark LinearAlgebra.mul!($y, $b, $x)
# b2 = @benchmark LinearAlgebra.mul!($y, $v, $x)
# b3 = @benchmark LinearAlgebra.mul!($y, $s, $x)
