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

"""
    buffers(::Type{T}, rows, cols) where {T}

Allocates and returns a set of buffers for efficient matrix-vector product computations.
The buffers are designed to minimize memory allocation and copying, reducing computational overhead.

# Arguments

  - `T`: The element type of the buffers.
  - `rows`: The number of rows in the matrix.
  - `cols`: The number of columns in the matrix.

# Returns

  - A tuple containing:

      + `forwardbuffer`: A view of the buffer for forward matrix-vector products, with length `rows`.
      + `adjointbuffer`: A view of the buffer for adjoint matrix-vector products, with length `cols`.
      + `buffer`: The underlying buffer, which is reused for both forward and adjoint products.

# Notes

  - The `buffer` is allocated with a length equal to the maximum of `rows` and `cols`, ensuring sufficient storage for both forward and adjoint products.
  - The `forwardbuffer` and `adjointbuffer` views are created using `unsafe_wrap`, which means that they overlap and share the same memory.
"""
function buffers(::Type{T}, rows, cols) where {T}
    buffer = Vector{T}(undef, max(rows, cols))
    forwardbuffer = unsafe_wrap(typeof(buffer), pointer(buffer), rows)
    adjointbuffer = unsafe_wrap(typeof(buffer), pointer(buffer), cols)
    return forwardbuffer, adjointbuffer, buffer
end

function Base.eltype(m::AbstractBlockMatrix{T}) where {T}
    return eltype(typeof(m))
end

function Base.eltype(::Type{<:AbstractBlockMatrix{T}}) where {T}
    return T
end

function Base.size(A::AbstractBlockMatrix)
    return A.size
end

"""
    buffer(A::AbstractBlockMatrix)

Returns the buffer associated with the given `AbstractBlockMatrix` instance for the
matrix-vector product.
"""
function buffer(A::AbstractBlockMatrix)
    return A.forwardbuffer
end

function buffer(
    A::M
) where {
    Z<:AbstractBlockMatrix,
    T,
    M<:Union{LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    return A.lmap.adjointbuffer
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector, α, β
) where {
    Z<:AbstractBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{<:Any,Z},LinearMaps.TransposeMap{<:Any,Z}},
}
    temp = buffer(A)
    mul!(temp, A, x)
    y .= β .* y
    y .+= α .* temp
    return y
end

"""
    rowblockids(A::AbstractBlockMatrix, i::Integer)

Returns the indices of the blocks in the `AbstractBlockMatrix` instance that contain the given row index `i`.

# Arguments

  - `A`: The `AbstractBlockMatrix` instance to query.
  - `i`: The row index to search for.

# Returns

  - A collection of block indices that contain the row index `i`.
"""
function rowblockids(A::AbstractBlockMatrix, i::Integer)
    return A.rowindexdict[i]
end

"""
    colblockids(A::AbstractBlockMatrix, j::Integer)

Returns the indices of the blocks in the `AbstractBlockMatrix` instance that contain the given column index `j`.

# Arguments

  - `A`: The `AbstractBlockMatrix` instance to query.
  - `j`: The column index to search for.

# Returns

  - A collection of block indices that contain the column index `j`.
"""
function colblockids(A::AbstractBlockMatrix, j::Integer)
    return A.colindexdict[j]
end

function Base.getindex(A::AbstractBlockMatrix, i::Integer, j::Integer)
    (i > size(A, 1) || j > size(A, 2)) && throw(BoundsError(A, (i, j)))

    for rowblockid in rowblockids(A, i)
        rowblockid ∉ colblockids(A, j) && continue
        b = block(A, rowblockid)
        I = findfirst(isequal(i), rowindices(b))
        isnothing(I) && continue
        J = findfirst(isequal(j), colindices(b))
        isnothing(J) && continue
        return b.matrix[I, J]
    end
    return zero(eltype(A))
end

function Base.setindex!(A::AbstractBlockMatrix, v, i::Integer, j::Integer)
    (i > size(A, 1) || j > size(A, 2)) && throw(BoundsError(A, (i, j)))

    for rowblockid in rowblockids(A, i)
        rowblockid ∉ colblockids(A, j) && continue
        b = block(A, rowblockid)
        I = findfirst(isequal(i), rowindices(b))
        isnothing(I) && continue
        J = findfirst(isequal(j), colindices(b))
        isnothing(J) && continue
        b.matrix[I, J] = v
        return b.matrix[I, J]
    end
    return error("Value not found")
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
