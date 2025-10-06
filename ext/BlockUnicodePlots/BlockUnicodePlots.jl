module BlockUnicodePlots
using BlockSparseMatrices
using UnicodePlots
using LinearMaps, SparseArrays

import BlockSparseMatrices: AbstractBlockMatrix, rowcolvals

function Base.show(
    io::IO, A::M
) where {
    Z<:AbstractBlockMatrix,
    T<:Number,
    M<:Union{Z,LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return println(io, blocksummary(io, A))
end

function blocksummary(
    io::IO, A::M
) where {
    Z<:AbstractBlockMatrix,
    T<:Number,
    M<:Union{Z,LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    rows, cols, vals = rowcolvals(A)
    return spy(
        size(A)...,
        rows,
        cols,
        abs.(vals);
        compact=true,
        border=:none,
        labels=false,
        title="$(LinearMaps.map_summary(A)) with $(nnz(A)) non-zero entries",
        maxheight=displaysize(io)[1] - 4,
        maxwidth=displaysize(io)[2] - 4,
    )
end

end # module BlockUnicodePlots
