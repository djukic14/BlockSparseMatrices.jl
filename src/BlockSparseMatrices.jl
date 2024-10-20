module BlockSparseMatrices
using LinearAlgebra
using SparseArrays
using LinearMaps

include("abstractblockmatrix.jl")
include("matrixblock/abstractmatrixblock.jl")
include("matrixblock/densematrixblock.jl")
include("blockmatrix.jl")
include("symmetricblockmatrix.jl")

export DenseMatrixBlock, BlockSparseMatrix, SymmetricBlockMatrix
export rowindices, colindices
export eachblockindex, block

end # module BlockSparseMatrices
