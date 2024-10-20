abstract type AbstractBlockMatrix{T} <: LinearMap{T} end

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
    T,
    Z<:AbstractBlockMatrix,
    M<:Union{Z,LinearMaps.AdjointMap{T,Z},LinearMaps.TransposeMap{T,Z}},
}
    temp = buffer(A)
    mul!(temp, A, x)
    y .= β .* y
    y .+= α .* temp
    return y
end
