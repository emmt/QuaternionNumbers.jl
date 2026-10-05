"""

Package `QuaternionNumbers` provides quaternions whose instances are `Number`s.

The package exports 3 types (`AbstractQuaternion{T} <: Number`, `Quaternion{T} <:
AbstractQuaternion{T}` and `Versor{T} <: AbstractQuaternion{T}`, with `T <: Real`
the type of the components of the quaternion) and 2 functions:

* `quaternion(args...)::Quaternion` builds a quaternion from arguments `args...`.

* `normalize(q::AbstractQuaternion)::Versor` yields the unit quaternion `q/abs(q)`.

The scalar and vector parts of a quaternion `q` are given by `real(q)` and `imag(q)`.

"""
module QuaternionNumbers

export AbstractQuaternion, Quaternion, Versor, quaternion, normalize

public 𝐢, 𝐣, 𝐤

using StaticArrays
using LinearAlgebra

const NVector{N} = Union{NTuple{N, Real}, SVector{N, <:Real}}

abstract type AbstractQuaternion{T <: Real} <: Number end

"""
    Quaternion(w, v = (0, 0, 0))
    Quaternion{T}(w, v = (0, 0, 0))
    Quaternion(w, x, y, z)
    Quaternion{T}(w, x, y, z)

Build a quaternion with scalar part `w` and 3-vector part `v = (x, y, z)`. Optional
parameter `T <: Real` is the type of the 4 values that constitute the quaternion.

The 4 components may also be specified as positional arguments or as a 4-tuple.

"""
struct Quaternion{T<:Real} <: AbstractQuaternion{T}
    vals::NTuple{4, T}
    Quaternion{T}(vals::NTuple{4, Real}) where {T <: Real} = new{T}(vals)
end

"""
    Versor(w, v = (0, 0, 0))
    Versor{T}(w, v = (0, 0, 0))

Build a unit quaternion with scalar part `w` and 3-vector part `v = (x, y, z)`. Optional
parameter `T <: Real` is the type of the 4 values that constitute the quaternion.

The 4 components may also be specified as positional arguments or as a 4-tuple.

The returned quaternion is [`normalize`d](@ref normalize) so that its norm is one.

"""
struct Versor{T<:Real} <: AbstractQuaternion{T}
    vals::NTuple{4, T}

    # Private inner constructor assuming without checking that arguments define a proper
    # unit quaternion.
    global _Versor
    _Versor(::Type{T}, vals::NTuple{4, Real}) where {T <: Real} = new{T}(vals)
end

const imx = _Versor(Bool, (false, true, false, false))
const imy = _Versor(Bool, (false, false, true, false))
const imz = _Versor(Bool, (false, false, false, true))
const 𝐢 = imx
const 𝐣 = imy
const 𝐤 = imz

constructor(q::AbstractQuaternion) = constructor(typeof(q))
constructor(::Type{<:Quaternion}) = Quaternion
constructor(::Type{<:Versor}) = Versor
constructor(::Type{<:AbstractQuaternion}) = AbstractQuaternion

Base.Tuple(q::AbstractQuaternion{T}) where {T} = getfield(q, 1)

Base.propertynames(q::AbstractQuaternion) = (:w, :x, :y, :z, :vals)
@inline Base.getproperty(q::AbstractQuaternion, key::Symbol) = _getproperty(q, Val(key))
_getproperty(q::AbstractQuaternion, ::Val{:w}) = Tuple(q)[1]
_getproperty(q::AbstractQuaternion, ::Val{:x}) = Tuple(q)[2]
_getproperty(q::AbstractQuaternion, ::Val{:y}) = Tuple(q)[3]
_getproperty(q::AbstractQuaternion, ::Val{:z}) = Tuple(q)[4]
_getproperty(q::AbstractQuaternion, ::Val{:vals}) = Tuple(q)
_getproperty(q::AbstractQuaternion, ::Val{key}) where {key} = throw(KeyError(key))

# Constructors with similar arguments and parameters.
for Q in (:Quaternion, :Versor)
    @eval begin
        # Positional arguments as tuple.
        $Q(vals::Vararg{Real, 4}) = $Q(vals)
        $Q{T}(vals::Vararg{Real, 4}) where {T <: Real} = $Q{T}(vals)

        # Copy constructors (more specific ones are defined elsewhere).
        $Q(q::$Q) = q
        $Q(q::AbstractQuaternion) = $Q(Tuple(q))
        $Q{T}(q::$Q{T}) where {T <: Real} = q

        # Build a quaternion given its scalar part.
        $Q(w::T) where {T <: Real} = $Q{T}(w)

        # Build a quaternion given its scalar and vector parts.
        $Q(w::Real, v::NTuple{3, Real}) = $Q(qtuple(w, v))
        $Q{T}(w::Real, v::NTuple{3}) where {T <: Real} = $Q{T}(qtuple(w, v))
        $Q(v::NTuple{3, Real}, w::Real, ) = $Q(qtuple(w, v))
        $Q{T}(v::NTuple{3}, w::Real) where {T <: Real} = $Q{T}(qtuple(w, v))

        # Build a quaternion from a 3- or 4-vector.
        $Q(v::AbstractVector{<: Real}) = $Q(qtuple(v))
        $Q{T}(v::AbstractVector{<: Real}) where {T <: Real} = $Q{T}(qtuple(v))
    end
end

# Constructors that call the inner constructor.
Quaternion(vals::NTuple{4, T}) where {T <: Real} = Quaternion{T}(vals)
Quaternion{T}(q::AbstractQuaternion) where {T <: Real} = Quaternion{T}(Tuple(q))
Versor{T}(q::Versor) where {T <: Real} = _Versor(T, Tuple(q))

# Promote arguments to the same type.
Quaternion(vals::NTuple{4, Real}) = Quaternion(promote(vals...))

# Building a unit quaternion implies normalization.
Versor(vals::NTuple{4, Real}) = _Versor(_normalize(vals))
Versor{T}(vals::NTuple{4, Real}) where {T <: Real} = _Versor(T, _normalize(vals))

_Versor(vals::NTuple{4, Real}) = _Versor(promote(vals...))
_Versor(vals::NTuple{4, T}) where {T <: Real} = _Versor(T, vals)

_normalize(vals::NTuple{4, Real}) = _normalize(promote(vals...))
_normalize(vals::NTuple{4, <: Real}) = _normalize(map(float, vals))
function _normalize((w,x,y,z)::NTuple{4, <: AbstractFloat}) # must be a homogeneous 4-tuple
    t = inv(hypot(w, x, y, z))
    isfinite(t) || throw(ArgumentError("cannot normalize quaternion"))
    return (t*w, t*x, t*y, t*z)
end

# Keyword-only constructors.
Quaternion(; w::Real, x::Real, y::Real, z::Real) = Quaternion(w, x, y, z)
function Quaternion{T}(; w::Real, x::Real, y::Real, z::Real) where {T <: Real}
    return Quaternion{T}(w, x, y, z)
end
Versor(; axis::NVector{3}, angle::Real) = _rotator(float, axis, angle)
function Versor{T}(; axis::NVector{3}, angle::Real) where {T <: AbstractFloat}
    return _rotator(Base.Fix1(convert, T), axis, angle)::Versor{T}
end
function _rotator(f, axis::NVector{3}, angle::Real)
    x, y, z, θ = map(f, promote(axis[1], axis[2], axis[3], angle))
    s, c = sincos(θ/2)
    t = s/sqrt(x^2 + y^2 + z^2)
    isfinite(t) || throw(ArgumentError("cannot normalize rotation axis"))
    return _Versor(c, (t*x, t*y, t*z))
end

# Implement abstract array API where it is not in contradiction with the API of numbers.

# TODO Base.eltype(q::AbstractQuaternion) = eltype(typeof(q))
# TODO Base.eltype(::Type{<:AbstractQuaternion{T}}) where {T<:Real} = T
# TODO
# TODO Base.length(q::AbstractQuaternion) = 4
# TODO Base.size(q::AbstractQuaternion) = (length(q),)
# TODO Base.axes(q::AbstractQuaternion) = (Base.axes1(q),)
# TODO Base.axes1(q::AbstractQuaternion) = Base.OneTo(4)
# TODO Base.IndexStyle(::Type{<:AbstractQuaternion}) = IndexLinear()

# Define `getindex` methods separately for each index type to avoid ambiguities.
for I in (Integer, AbstractRange{<:Integer}, AbstractArray{Bool})
    @eval @inline Base.getindex(q::AbstractQuaternion, i::$I) = getindex(Tuple(q), i)
end

Base.iterate(::AbstractQuaternion) = error(
    "Iteration is deliberately unsupported for quaternions. Use `Tuple(q)...`")

# Conversion constructors.
AbstractQuaternion(q::AbstractQuaternion) = q
AbstractQuaternion{T}(q::AbstractQuaternion{T}) where {T<:Real} = q
AbstractQuaternion{T}(q::AbstractQuaternion) where {T<:Real} = constructor(q){T}(q)

# Convert a scalar into a quaternion. TODO Same thing for a 3-vector.
AbstractQuaternion(w::Real) = Quaternion(w)
AbstractQuaternion{T}(w::Real) where {T <: Real} = Quaternion{T}(w)
Quaternion{T}(w::Real) where {T <: Real} = Quaternion{T}((w, default_vector_part(T)...))
function Versor{T}(w::Real) where {T <: Real}
    u = isnegative(w) ? -one(T) : one(T)
    return _Versor(T, (u, default_vector_part(T)...))
end

function default_vector_part(::Type{T}) where {T <: Real}
    z = zero(T) # save memory for types such as big numbers
    return (z, z, z)
end
#default_scalar_part(::Type{NTuple{3, T}}) where {T <: Real} = zero(T)
#default_scalar_part(::Type{NTuple{3, Real}}) = false

to_tuple(t::Tuple) = t
to_tuple(v::SVector) = Tuple(v)
#to_tuple(v::AbstractVector) = Tuple(v)

"""
    real(q::AbstractQuaternion) -> w::Real

Return the real part (also called scalar part) of the quaternion `q`. The result is a real.

"""
Base.real(q::AbstractQuaternion) = q.w

"""
    real(::Type{<:AbstractQuaternion{T}}) -> T

Return the type of the real part of a quaternion given its type. The other components of the
quaternion have the same type.

"""
Base.real(q::Type{<:AbstractQuaternion{T}}) where {T <: Real} = T

"""
    imag(q::AbstractQuaternion) -> (x,y,z)::NTuple{3, Real}

Return the imaginary part (also called vector part) of the quaternion `q`. The result is a
3-tuple of reals.

"""
Base.imag(q::AbstractQuaternion) = Tuple(q)[2:4]

Base.isreal(q::AbstractQuaternion) = (iszero(q.x) & iszero(q.y) & iszero(q.z))
Base.isinteger(q::AbstractQuaternion) = isreal(q) & isinteger(real(q))
Base.iszero(q::AbstractQuaternion) = isreal(q) & iszero(real(q))
Base.isone(q::AbstractQuaternion) = isreal(q) & isone(real(q))
Base.isfinite(q::AbstractQuaternion) = qisfinite(Tuple(q))
Base.isnan(q::AbstractQuaternion) = qisnan(Tuple(q))
Base.isinf(q::AbstractQuaternion) = qisinf(Tuple(q))

Base.one(q::AbstractQuaternion) = one(typeof(q))
Base.one(::Type{<:AbstractQuaternion}) = one(AbstractQuaternion{Bool})
function Base.one(::Type{<:AbstractQuaternion{T}}) where {T <: Real}
    return _Versor(T, (one(T), default_vector_part(T)...))
end

Base.zero(q::AbstractQuaternion) = zero(typeof(q))
Base.zero(::Type{<:AbstractQuaternion}) = zero(AbstractQuaternion{Bool})
function Base.zero(::Type{<:AbstractQuaternion{T}}) where {T <: Real}
    z = zero(T) # save memory for types such as big numbers
    return Quaternion{T}((z, z, z, z))
end

"""
    quaternion(w, x, y, z)
    quaternion(w, v = (0,0,0))
    quaternion((w, x, y, z))

Return a quaternion with scalar part `w` and vector part `(x,y,z)`. The vector part may
also be specified by a 3-tuple `v = (x,y,z)` or by a 3-vector. All components may also
be specified by the 4-tuple `q = (w, x, y, z)` or by a 4-vector.

The scalar and vector part are also called the *real* and *imaginary* part of the
quaternion.

"""
quaternion(w::Real) = Quaternion(w)
quaternion(q::Vararg{Real, 4}) = Quaternion(q)
quaternion(q::NTuple{4, Real}) = Quaternion(q)
quaternion(q::SVector{4, <: Real}) = Quaternion(Tuple(sijk))
quaternion(w::Real, v::NTuple{3, Real}) = Quaternion(w, v)
quaternion(w::Real, v::SVector{3, <: Real}) = Quaternion(w, Tuple(v))

"""
    LinearAlgebra.normalize(q::AbstractQuaternion) -> u::Versor

Return the unit quaternion `q/abs(q)`.

"""
LinearAlgebra.normalize(q::Versor) = q
LinearAlgebra.normalize(q::AbstractQuaternion) = Versor(Tuple(q))

Base.write(io::IO, q::AbstractQuaternion) = write(io, Tuple(q)...)
function Base.read(io::IO, ::Type{Quaternion{T}}) where {T <: Real}
    return Quaternion{T}(read(io, T), read(io, T), read(io, T), read(io, T))
end

# Byte order swaps: components are swapped individually.
Base.bswap(q::Quaternion) = Quaternion(qmap(bswap, Tuple(q)))
Base.bswap(q::Versor) = _Versor(qmap(bswap, Tuple(q)))

# TODO hash

# Unary minus.
Base.:(-)(q::Quaternion) = Quaternion(map(-, Tuple(q)))
Base.:(-)(q::Versor{T}) where {T <: Real} = _Versor(map(-, Tuple(q)))

# Addition (commutative).
Base.:(+)(q::AbstractQuaternion, r::Real) = Quaternion(qcall(qadd, Tuple(q), r))
Base.:(+)(r::Real, q::AbstractQuaternion) = q + r
Base.:(+)(q::AbstractQuaternion, v::NVector{3}) = Quaternion(qcall(qadd, Tuple(q), Tuple(v)))
Base.:(+)(v::NVector{3}, q::AbstractQuaternion) = q + v
function Base.:(+)(a::AbstractQuaternion, b::AbstractQuaternion)
    return Quaternion(qcall(qadd, Tuple(a), Tuple(b)))
end

# Subtraction.
Base.:(-)(q::AbstractQuaternion, r::Real) = r * q
Base.:(-)(r::Real, q::AbstractQuaternion) = Quaternion(qcall(qsub, r, Tuple(q)))
Base.:(-)(q::AbstractQuaternion, v::NVector{3}) = Quaternion(qcall(qsub, Tuple(q), Tuple(v)))
Base.:(-)(v::NVector{3}, q::AbstractQuaternion) = Quaternion(qcall(qsub, Tuple(v), Tuple(q)))
function Base.:(-)(a::AbstractQuaternion, b::AbstractQuaternion)
    return Quaternion(qcall(qsub, Tuple(a), Tuple(b)))
end

# Multiplication (non-commutative except with a scalar).
Base.:(*)(q::AbstractQuaternion, r::Real) = r * q
Base.:(*)(r::Real, q::AbstractQuaternion) = Quaternion(qcall(qmul, r, Tuple(q)))
Base.:(*)(q::AbstractQuaternion, v::NVector{3}) = Quaternion(qcall(qmul, Tuple(q), Tuple(v)))
Base.:(*)(v::NVector{3}, q::AbstractQuaternion) = Quaternion(qcall(qmul, Tuple(v), Tuple(q)))
Base.:(*)(a::Versor, b::Versor) = _Versor(qcall(qmul, Tuple(a), Tuple(b)))
function Base.:(*)(a::AbstractQuaternion, b::AbstractQuaternion)
    return Quaternion(qcall(qmul, Tuple(a), Tuple(b)))
end

# Division.
Base.:(\)(r::Real, q::AbstractQuaternion) = q / r
Base.:(/)(q::AbstractQuaternion, r::Real) = Quaternion(qcall(qdiv, Tuple(q), r))
function Base.:(\)(a::AbstractQuaternion, b::AbstractQuaternion)
    return Quaternion(qcall(qldiv, Tuple(a), Tuple(b)))
end
function Base.:(/)(a::AbstractQuaternion, b::AbstractQuaternion)
    return Quaternion(qcall(qrdiv, Tuple(a), Tuple(b)))
end
# TODO other division

# Conjugation, absolute value, norms, inverse, etc.
for (Q, f) in (AbstractQuaternion => Quaternion,
               Versor => _Versor)
    @eval Base.conj(q::$Q) = $f((q.w, -q.x, -q.y, -q.z))
end

LinearAlgebra.norm(q::AbstractQuaternion) = abs(q)
Base.abs(q::Versor{T}) where {T} = one(T)
Base.abs(q::AbstractQuaternion) = qabs(Tuple(q))

Base.abs2(q::Versor{T}) where {T} = one(T)
Base.abs2(q::AbstractQuaternion) = qabs2(Tuple(q))

Base.sign(q::Versor) = q
function Base.sign(q::Quaternion)
    a = float(abs(q))
    return (iszero(a) ? float(q) : q/a)::Quaternion{typeof(a)}
end

Base.inv(q::Versor) = conj(q)
Base.inv(q::AbstractQuaternion) = Quaternion(qinv(Tuple(q)))

# Maximal absolute value of components.
@inline norminf(q::AbstractQuaternion) = qnorminf(Tuple(q))

# TODO muladd
# TODO pow (^)

# `AbstractFloat(q)` is also used by `float(q)`.
(::Type{AbstractFloat})(q::AbstractQuaternion{T}) where {T <: AbstractFloat} = q
(::Type{AbstractFloat})(q::AbstractQuaternion{T}) where {T <: Real} =
    constructor(q){float(T)}(q)

Base.big(q::AbstractQuaternion{T}) where {T <: Real} = constructor(q){big(T)}(q)
Base.big(::Type{Q}) where {T, Q <: AbstractQuaternion{T}} = constructor(Q){big(T)}

# Machine epsilon for quaternions
Base.eps(q::AbstractQuaternion{<:AbstractFloat}) = hypot(map(eps, Tuple(q))...)
Base.eps(::Type{<:AbstractQuaternion{T}}) where {T<:AbstractFloat} = 2*eps(T) # 2 is sqrt(4)

for f in (:(==), :isequal)
    @eval begin
        function Base.$f(a::AbstractQuaternion, b::AbstractQuaternion)
            return qmapfoldl($f, &, Tuple(a), Tuple(b))
        end
        Base.$f(r::Number, q::AbstractQuaternion) = $f(q, r)
        Base.$f(q::AbstractQuaternion, r::Number) = $f(real(q), r) & isreal(q)
    end
end

function Base.promote_rule(::Type{<:AbstractQuaternion{S}},
                           ::Type{T}) where {S <: Real, T <: Real}
    return Quaternion{promote_type(S, T)}
end
function Base.promote_rule(::Type{<:AbstractQuaternion{S}},
                           ::Type{<:AbstractQuaternion{T}}) where {S <: Real, T <: Real}
    return Quaternion{promote_type(S, T)}
end

#--------------------------------------------------------------------- Low level functions -

@inline qmap(f, q::NTuple{4, <: Real}) = (f(q[1]), f(q[2]), f(q[3]), f(q[4]))
@inline qmap(f, a::NTuple{4, <: Real}, b::NTuple{4, <: Real}) =
    (f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3]), f(a[4], b[4]))

@inline qmapfoldl(f, op, q::NTuple{4, <: Real}) =
    op(op(op(f(q[1]), f(q[2])), f(q[3])), f(q[4]))
@inline qmapfoldl(f, op, a::NTuple{4, <: Real}, b::NTuple{4, <: Real}) =
    op(op(op(f(a[1], b[1]), f(a[2], b[2])), f(a[3], b[3])), f(a[4], b[4]))

@inline qmapfoldr(f, op, q::NTuple{4, <: Real}) =
    op(f(q[1]), op(f(q[2]), op(f(q[3]), f(q[4]))))
@inline qmapfoldr(f, op, a::NTuple{4, <: Real}, b::NTuple{4, <: Real}) =
    op(f(a[1], b[1]), op(f(a[2], b[2]), op(f(a[3], b[3]), f(a[4], b[4]))))

qvec(q::NTuple{4, <: Real}) = Tuple(q)[2:4]
qdot(a::NVector{3}, b::NVector{3}) = mapreduce(*, +, a, b)
qcross(a::NVector{3}, b::NVector{3}) = (a[2]*b[3] - a[3]*b[2],
                                        a[3]*b[1] - a[1]*b[3],
                                        a[1]*b[2] - a[2]*b[1])

@inline qadd(q::NTuple{4, T}, r::T) where {T <: Real} = (q[1] + r, q[2], q[3], q[4])
@inline function qadd(q::NTuple{4, T}, v::NTuple{3, T}) where {T <: Real}
    return (q[1], q[2] + v[1], q[3] + v[2], q[4] + v[3])
end
@inline function qadd(a::NTuple{4, T}, b::NTuple{4, T}) where {T <: Real}
    return (a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])
end

@inline qsub(q::NTuple{4, T}, r::T) where {T <: Real} = (q[1] - r, q[2], q[3], q[4])
@inline qsub(r::T, q::NTuple{4, T}) where {T <: Real} = (r - q[1], -q[2], -q[3], -q[4])
@inline function qsub(q::NTuple{4, T}, v::NTuple{3, T}) where {T <: Real}
    return (q[1], q[2] - v[1], q[3] - v[2], q[4] - v[3])
end
@inline function qsub(v::NTuple{3, T}, q::NTuple{4, T}) where {T <: Real}
    return (-q[1], v[1] - q[2], v[2] - q[3], v[3] - q[4])
end
@inline function qsub(a::NTuple{4, T}, b::NTuple{4, T}) where {T <: Real}
    return (a[1] - b[1], a[2] - b[2], a[3] - b[3], a[4] - b[4])
end

@inline qmul(r::T, q::NTuple{4, T}) where {T <: Real} = (q[1]*r, q[2]*r, q[3]*r, q[4]*r)
@inline function qmul((a_w, a_x, a_y, a_z)::NTuple{4, T},
                      (b_w, b_x, b_y, b_z)::NTuple{4, T}) where {T<:Real}
    # Hamilton product (28 flops)
    return ((a_w * b_w - a_x * b_x) - (a_y * b_y + a_z * b_z),
            (a_w * b_x + a_x * b_w) + (a_y * b_z - a_z * b_y),
            (a_w * b_y + a_y * b_w) + (a_z * b_x - a_x * b_z),
            (a_w * b_z + a_z * b_w) + (a_x * b_y - a_y * b_x))
end
@inline function qmul((     a_x, a_y, a_z)::NTuple{3, T},
                      (b_w, b_x, b_y, b_z)::NTuple{4, T}) where {T<:Real}
    return (-a_x * b_x - a_y * b_y - a_z * b_z,
            a_x * b_w + (a_y * b_z - a_z * b_y),
            a_y * b_w + (a_z * b_x - a_x * b_z),
            a_z * b_w + (a_x * b_y - a_y * b_x))
end
@inline function qmul((a_w, a_x, a_y, a_z)::NTuple{4, T},
                      (     b_x, b_y, b_z)::NTuple{3, T}) where {T<:Real}
    return (-a_x * b_x - a_y * b_y - a_z * b_z,
            a_w * b_x + (a_y * b_z - a_z * b_y),
            a_w * b_y + (a_z * b_x - a_x * b_z),
            a_w * b_z + (a_x * b_y - a_y * b_x))
end
@inline function qmul((a_x, a_y, a_z)::NTuple{3, T},
                      (b_x, b_y, b_z)::NTuple{3, T}) where {T<:Real}
    return (-a_x * b_x - a_y * b_y - a_z * b_z,
            a_y * b_z - a_z * b_y,
            a_z * b_x - a_x * b_z,
            a_x * b_y - a_y * b_x)
end

@inline function qinv(q::NTuple{4, <: Real})
    # TODO avoid overflows
    r = float(qabs2(q))
    t = -r
    return Quaternion(q.w/r, q.x/t, q.y/t, q.z/t)
end

@inline qdiv(q::NTuple{4, T}, r::T) where {T <: Real} = (q[1]/r, q[2]/r, q[3]/r, q[4]/r)

# Compute the infinite norm (maximal absolute value) of a 4-tuple without taking care of
# propagating NaNs (like with `@fastmath`).
@inline fast_qnorminf(q::NTuple{4, <: Real}) = fast_qnorminf(q...)
function fast_qnorminf(q1::T, q2::T, q3::T, q4::T) where {T <: Real}
    a1 = abs(q1)
    a2 = abs(q2)
    a3 = abs(q3)
    a4 = abs(q4)
    a12 = a1 < a2 ? a2 : a1
    a34 = a3 < a4 ? a4 : a3
    return a12 < a34 ? a34 : a12
end

# Return maximal absolute value.
function qnorminf(q::NTuple{4, <: Real})
    # Propagate NaNs but avoid branching for faster operation.
    a1 = abs(q[1])
    a2 = abs(q[2])
    a3 = abs(q[3])
    a4 = abs(q[4])
    a12 = (isnan(a2) | (a1 < a2)) ? a2 : a1
    a34 = (isnan(a4) | (a3 < a4)) ? a4 : a3
    return (isnan(a34) | (a12 < a34)) ? a34 : a12
end


# Return Euclidean norm.
# TODO Check whether, NaN and Inf are properly returned
function qabs(q::NTuple{4, <: Real})
    # Normalize by maximal absolute value to avoid overflows.
    r = float(qnorminf(q))
    iszero(r) && return r
    isfinite(r) || return oftype(r, (qisinf(q) ? Inf : NaN))
    return r*sqrt(abs2(q[1]/r) + abs2(q[2]/r) + abs2(q[3]/r) + abs2(q[4]/r))
end

qabs2(q::NTuple{4, <: Real}) = qmapfoldl(abs2, +, q)

qisfinite(q::NTuple{4, <: Integer}) = true
qisfinite(q::NTuple{4, <: Real}) = isfinite(q[1]) & isfinite(q[3]) & isfinite(q[3]) & isfinite(q[4])
qisfinite1(q::NTuple{4, <: Real}) = qmapfoldl(isfinite, &, q)

qisinf(q::NTuple{4, <: Integer}) = false
qisinf(q::NTuple{4, <: Real}) = isinf(q[1]) | isinf(q[3]) | isinf(q[3]) | isinf(q[4])

qisnan(q::NTuple{4, <: Integer}) = false
qisnan(q::NTuple{4, <: Real}) = isnan(q[1]) | isnan(q[3]) | isnan(q[3]) | isnan(q[4])

# Implement `a\b`, the left-division of `b` by `a`.
function qldiv(a::NTuple{4, T}, b::NTuple{4, T}) where {T<:Real}
    # Normalize by the Euclidean norm of `a` to avoid overflows.
    r = float(qabs(a))
    return qdiv(qmul(qconj(qdiv(a, r)), b), r)
end

# Implement `a/b`, the right-division of `a` by `b`.
function qrdiv(a::NTuple{4, T}, b::NTuple{4, T}) where {T<:Real}
    # Normalize by the Euclidean norm of `b` to avoid overflows.
    r = float(qabs(b))
    return qdiv(qmul(a, qconj(qdiv(b, r))), r)
end

# `qcall(f, x, y)` calls the function `f` with promoted arguments `x` and `y`.

@inline qcall(f, q::NTuple{4, T}, r::T) where {T <: Real} = f(q, r)
@inline function qcall(f, q::NTuple{4, Real}, r::Real)
    _q1, _q2, _q3, _q4, _r = promote(q..., r)
    return f((_q1, _q2, _q3, _q4), _r)
end

@inline qcall(f, r::T, q::NTuple{4, T}) where {T <: Real} = f(r, q)
@inline function qcall(f, r::Real, q::NTuple{4, Real})
    _q1, _q2, _q3, _q4, _r = promote(q..., r)
    return f(_r, (_q1, _q2, _q3, _q4))
end

@inline qcall(f, q::NTuple{4, T}, v::NTuple{3, T}) where {T <: Real} = f(q, v)
@inline function qcall(f, q::NTuple{4, Real}, v::NTuple{3, Real})
    _q1, _q2, _q3, _q4, _v1, _v2, _v3 = promote(q..., v...)
    return f((_q1, _q2, _q3, _q4), (_v1, _v2, _v3))
end

@inline qcall(f, v::NTuple{3, T}, q::NTuple{4, T}) where {T <: Real} = f(v, q)
@inline function qcall(f, v::NTuple{3, Real}, q::NTuple{4, Real})
    _q1, _q2, _q3, _q4, _v1, _v2, _v3 = promote(q..., v...)
    return f((_v1, _v2, _v3), (_q1, _q2, _q3, _q4))
end

@inline qcall(f, a::NTuple{4, T}, b::NTuple{4, T}) where {T <: Real} = f(a, b)
@inline function qcall(f, a::NTuple{4, Real}, b::NTuple{4, Real})
    _a1, _a2, _a3, _a4, _b1, _b2, _b3, _b4 = promote(a..., b...)
    return f((_a1, _a2, _a3, _a4), (_b1, _b2, _b3, _b4))
end
# Return a 4-tuple of reals from given arguments. Result may not be homogeneous.
qtuple(t::NTuple{3, Real}) = (false, t...)
qtuple(t::NTuple{4, Real}) = t
qtuple(t::NTuple{N, Real}) where {N} = throw_invalid_quaternion_length(N)
#
qtuple(r::Real, t::NTuple{3, Real}) = (r, t...)
qtuple(t::NTuple{N, Real}, r::Real) where {N} = qtuple(r, t)
qtuple(r::Real, t::NTuple{N, Real}) where {N} = throw_invalid_vector_part_length(N)
#
qtuple(v::SVector{N, Real}) where {N} = qtuple(Tuple(v))
qtuple(r::Real, v::SVector{N, Real}) where {N} = qtuple(r, Tuple(v))
qtuple(v::SVector{N, Real}, r::Real) where {N} = qtuple(r, v)
#
function qtuple(v::AbstractVector{T}) where {T <: Real}
    n = length(v)
    3 ≤ n ≤ 4 || throw_invalid_quaternion_length(n)
    @inbounds begin
        i = firstindex(v)
        n == 4 && return (v[i], v[i+1], v[i+2], v[i+3])::NTuple{4, T}
        return (zero(T), v[i], v[i+1], v[i+2])::NTuple{4, T}
    end
end
qtuple(v::AbstractVector{<: Real}, r::Real) = qtuple(r, v)
function qtuple(r::Real, v::AbstractVector{<: Real})
    n = length(v)
    n == 3 || throw_invalid_vector_part_length(n)
    @inbounds  begin
        i = firstindex(v)
        return (r, v[i], v[i+1], v[i+2])
    end
end

@noinline throw_invalid_vector_part_length(n::Integer) = throw(DimensionMismatch(
    "vector part of quaternion must have 3 entries, got $n"))

@noinline throw_invalid_quaternion_length(n::Integer) = throw(DimensionMismatch(
    "vector/tuple must have 3 or 4 entries to build a quaternion, got $n"))

#--------------------------------------------------------------------------- Compatibility -

# `ispositive` and `isnegative` are not defined in all Julia versions.
if !isdefined(Base, :ispositive)
    ispositive(x::Real) = x > 0
    ispositive(x::Unsigned) = !iszero(x)
    ispositive(x::Bool) = x
end
if !isdefined(Base, :isnegative)
    isnegative(x::Real) = x < 0
    isnegative(x::Unsigned) = false
    isnegative(x::Bool) = false
end

end
