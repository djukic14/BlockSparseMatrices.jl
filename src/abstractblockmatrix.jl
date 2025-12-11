"""
    abstract type AbstractBlockMatrix{T} <: LinearMap{T}

Abstract type representing a block matrix with element type `T`.
The block matrix is composed of a small number of smaller matrix blocks, allowing for
efficient storage and computation.

# Notes

  - The `AbstractBlockMatrix` type serves as a base for concrete block matrix implementations,
    providing a common interface for linear algebra operations.
"""
abstract type AbstractBlockMatrix{T} <: LinearMap{T} end

function Base.eltype(m::AbstractBlockMatrix{T}) where {T}
    return eltype(typeof(m))
end

function Base.eltype(::Type{<:AbstractBlockMatrix{T}}) where {T}
    return T
end

function Base.size(A::AbstractBlockMatrix)
    return A.size
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {
    Z<:AbstractBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    return LinearMaps._unsafe_mul!(y, A, x, true, false)
end

"""
    scheduler(A::AbstractBlockMatrix)

Returns the scheduler associated with the given `AbstractBlockMatrix` instance.
This scheduler is responsible for managing the parallel computation of matrix-vector products.

# Returns

  - The scheduler associated with the matrix.

# Notes

  - The scheduler is used to coordinate the computation of matrix-vector product.
"""
function scheduler(A::AbstractBlockMatrix)
    return A.scheduler
end

function scheduler(
    A::M
) where {
    Z<:AbstractBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return scheduler(A.lmap)
end

function _nnz(A)
    return SparseArrays.nnz(A)
end

function _nnz(A::AbstractArray)
    return prod(size(A))
end
