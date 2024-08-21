module BlockSparseMatrices
using LinearAlgebra
using SparseArrays
using LinearMaps

include("matrixblock/abstractmatrixblock.jl")
include("matrixblock/densematrixblock.jl")
include("blockmatrix.jl")

export DenseMatrixBlock, BlockSparseMatrix
export rowindices, colindices
export eachblockindex, block

end # module BlockSparseMatrices
