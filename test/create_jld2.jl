using Test
using BlockSparseMatrices
using BEAST, CompScienceMeshes
using H2Trees
using SparseArrays
using OhMyThreads
using JLD2

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

# m = meshsphere(1.0, 0.1)
m = meshcuboid(1.0, 1.0, 1.0, 0.04)
X = raviartthomas(m)
tree = TwoNTree(X, 0.05)
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

# d = Dict(zip(["sphere"], [(blocks, testindices, trialindices)]))

jldsave("vbcrsexample.jld2"; blockdict=d)
