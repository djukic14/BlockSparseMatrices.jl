module BlockSparseMatrices
using LinearAlgebra
using SparseArrays
using LinearMaps
using OhMyThreads
using Graphs, GraphsColoring

import GraphsColoring: conflicts, conflictgraph, conflictmatrix, ConflictFunctor

const coloringalgorithm = WorkstreamDSATUR

function isserial(::Scheduler)
    return false
end

function isserial(::SerialScheduler)
    return true
end

include("abstractblockmatrix.jl")
include("matrixblock/abstractmatrixblock.jl")
include("matrixblock/densematrixblock.jl")
include("blockmatrix.jl")
include("symmetricblockmatrix.jl")

export DenseMatrixBlock, BlockSparseMatrix, SymmetricBlockMatrix
export rowindices, colindices
export eachblockindex, block

include("sparse.jl")

# for backwards compatibility with julia versions below 1.9
if !isdefined(Base, :get_extension)
    include("../ext/BlockUnicodePlots/BlockUnicodePlots.jl")
end

end # module BlockSparseMatrices
