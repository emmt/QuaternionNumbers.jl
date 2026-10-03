"""

Package `QuaternionNumbers` provides quaternions whose instances are `Number`s.

The package exports 3 types (`AbstractQuaternion{T} <: Number`, `Quaternion{T} <:
AbstractQuaternion{T}` and `UnitQuaternion{T} <: AbstractQuaternion{T}`, with `T <: Real`
the type of the components of the quaternion) and 2 functions:

* `quaternion(args...)::Quaternion` builds a quaternion from arguments `args...`.

* `normalize(q::AbstractQuaternion)::UnitQuaternion` yields the unit quaternion `q/abs(q)`.

The scalar and vector parts of a quaternion `q` are given by `real(q)` and `imag(q)`.

"""
module QuaternionNumbers

export AbstractQuaternion, Quaternion, UnitQuaternion, quaternion, normalize

public 𝐢, 𝐣, 𝐤

using StaticArrays
using LinearAlgebra

const VectorLike{N,T} = Union{NTuple{N, T}, SVector{N, <: T}}

abstract type AbstractQuaternion{T<:Real} <: Number end

"""
    Quaternion(s, (i, j, k))
    Quaternion{T}(s, (i, j, k))

Build a quaternion with scalar part `s` and vector part `(i, j, k)`. Optional parameter `T
<: Real` is the type of the 4 values that constitute the quaternion.

The 4 components may also be specified as positional arguments or as a 4-tuple.

"""
struct Quaternion{T<:Real} <: AbstractQuaternion{T}
    vals::NTuple{4, T}
    Quaternion{T}(vals::NTuple{4, Real}) where {T <: Real} = new{T}(vals)
end

"""
    UnitQuaternion(s, (i, j, k))
    UnitQuaternion{T}(s, (i, j, k))

Build a unit quaternion with scalar part `s` and vector part `(i, j, k)`. Optional parameter
`T <: Real` is the type of the 4 values that constitute the quaternion.

The 4 components may also be specified as positional arguments or as a 4-tuple.

The returned quaternion is [`normalize`d](@ref normalize) so that its norm is one.

"""
struct UnitQuaternion{T<:Real} <: AbstractQuaternion{T}
    vals::NTuple{4, T}

    # Private inner constructor assuming without checking that arguments define a proper
    # unit quaternion.
    global _UnitQuaternion
    _UnitQuaternion(::Type{T}, vals::NTuple{4, Real}) where {T <: Real} = new{T}(vals)
end

constructor(q::AbstractQuaternion) = constructor(typeof(q))
constructor(::Type{<:Quaternion}) = Quaternion
constructor(::Type{<:UnitQuaternion}) = UnitQuaternion
constructor(::Type{<:AbstractQuaternion}) = AbstractQuaternion

Base.Tuple(q::AbstractQuaternion{T}) where {T} = getfield(q, 1)

Base.propertynames(q::AbstractQuaternion) = (:s, :i, :j, :k, :vals)
@inline Base.getproperty(q::AbstractQuaternion, key::Symbol) = _getproperty(q, Val(key))
_getproperty(q::AbstractQuaternion, ::Val{:s}) = Tuple(q)[1]
_getproperty(q::AbstractQuaternion, ::Val{:i}) = Tuple(q)[2]
_getproperty(q::AbstractQuaternion, ::Val{:j}) = Tuple(q)[3]
_getproperty(q::AbstractQuaternion, ::Val{:k}) = Tuple(q)[4]
_getproperty(q::AbstractQuaternion, ::Val{:vals}) = Tuple(q)
_getproperty(q::AbstractQuaternion, ::Val{key}) where {key} = throw(KeyError(key))

# Constructors with similar arguments and parameters.
for Q in (:Quaternion, :UnitQuaternion)
    @eval begin
        # Positional arguments as tuple.
        $Q(vals::Vararg{Real, 4}) = $Q(vals)
        $Q{T}(vals::Vararg{Real, 4}) where {T} = $Q{T}(vals)

        # Copy constructors (more specific ones are defined elsewhere).
        $Q(q::$Q) = q
        $Q(q::AbstractQuaternion) = $Q(Tuple(q))
        $Q{T}(q::$Q{T}) where {T<:Real} = q
    end
end

# Constructors that call the inner constructor.
Quaternion(vals::NTuple{4, T}) where {T <: Real} = Quaternion{T}(vals)
Quaternion{T}(q::AbstractQuaternion) where {T <: Real} = Quaternion{T}(Tuple(q))
UnitQuaternion{T}(q::UnitQuaternion) where {T <: Real} = _UnitQuaternion(T, Tuple(q))

# Promote arguments to the same type.
Quaternion(vals::NTuple{4, Real}) = Quaternion(promote(vals...))

# Building a unit quaternion implies normalization.
UnitQuaternion(vals::NTuple{4, Real}) = _UnitQuaternion(_normalize(vals))
function UnitQuaternion{T}(vals::NTuple{4, Real}) where {T <: Real}
    return _UnitQuaternion(T, _normalize(vals))
end

_normalize(vals::NTuple{4, Real}) = _normalize(promote(vals...))
_normalize(vals::NTuple{4, <:Real}) = _normalize(map(float, vals))
function _normalize((s,i,j,k)::NTuple{4, <: AbstractFloat}) # must be a homogeneous 4-tuple
    t = inv(hypot(s, i, j, k))
    isfinite(t) || throw(ArgumentError("cannot normalize quaternion"))
    return (t*s, t*i, t*j, t*k)
end

_UnitQuaternion(vals::NTuple{4, Real}) = _UnitQuaternion(promote(vals...))
_UnitQuaternion(vals::NTuple{4, T}) where {T <: Real} = _UnitQuaternion(T, vals)

# Keyword-only constructors.
Quaternion(; s::Real, i::Real, j::Real, k::Real) = Quaternion(s, i, j, k)
function Quaternion{T}(; s::Real, i::Real, j::Real, k::Real) where {T <: Real}
    return Quaternion{T}(s, i, j, k)
end
UnitQuaternion(; axis::Vector3D, angle::Real) = _rotator(float, axis, angle)
function UnitQuaternion{T}(; axis::Vector3D, angle::Real) where {T <: AbstractFloat}
    return _rotator(Base.Fix1(convert, T), axis, angle)::UnitQuaternion{T}
end
function _rotator(f, axis::Vector3D, angle::Real)
    x, y, z, θ = map(f, promote(axis[1], axis[2], axis[3], angle))
    s, c = sincos(θ/2)
    t = s/sqrt(x^2 + y^2 + z^2)
    isfinite(t) || throw(ArgumentError("cannot normalize rotation axis"))
    return _UnitQuaternion(c, (t*x, t*y, t*z))
end

# Implement abstract array API where it is not in contradiction with the API of numbers.
Base.eltype(q::AbstractQuaternion) = eltype(typeof(q))
Base.eltype(::Type{<:AbstractQuaternion{T}}) where {T<:Real} = T

Base.length(q::AbstractQuaternion) = 4
Base.size(q::AbstractQuaternion) = (length(q),)
Base.axes(q::AbstractQuaternion) = (Base.axes1(q),)
Base.axes1(q::AbstractQuaternion) = Base.OneTo(4)
Base.IndexStyle(::Type{<:AbstractQuaternion}) = IndexLinear()

const Index = Union{Integer,}

# Define `getindex` methods separately to avoid ambiguities.
for I in (Integer, AbstractRange{<:Integer}, AbstractArray{Bool})
    @eval @inline Base.getindex(q::AbstractQuaternion, i::$I) = getindex(Tuple(q), i)
end

Base.iterate(::AbstractQuaternion) =
    error("iteration is deliberately unsupported for quaternions. Use `Tuple(q)...`")

# Conversion constructors.
AbstractQuaternion(q::AbstractQuaternion) = q
AbstractQuaternion{T}(q::AbstractQuaternion{T}) where {T<:Real} = q
AbstractQuaternion{T}(q::AbstractQuaternion) where {T<:Real} = constructor(q){T}(q)

# Convert a scalar into a quaternion. TODO Same thing for a 3-vector.
AbstractQuaternion(s::Real) = Quaternion(s)
AbstractQuaternion{T}(s::Real) where {T <: Real} = Quaternion{T}(s)
function Quaternion(s::Real)
    z = zero(typeof(s)) # save memory for types such as big numbers
    return Quaternion((s, z, z, z))
end
function Quaternion{T}(s::Real) where {T <: Real}
    z = zero(T) # save memory for types such as big numbers
    return Quaternion{T}((convert(T, s), z, z, z))
end
function UnitQuaternion(s::Real)
    u = isnegative(s) ? -one(s) : one(s)
    z = zero(typeof(s)) # save memory for types such as big numbers
    return _UnitQuaternion((u, z, z, z))
end
function UnitQuaternion{T}(s::Real) where {T <: Real}
    u = isnegative(s) ? -one(T) : one(T) # TODO check that T is signed
    z = zero(T) # save memory for types such as big numbers
    return _UnitQuaternion(T, (u, z, z, z))
end

"""
    real(q::AbstractQuaternion) -> s::Real

Return the real part (also called scalar part) of the quaternion `q`. The result is a real.

"""
Base.real(q::AbstractQuaternion) = q.s

"""
    imag(q::AbstractQuaternion) -> ijk::NTuple{3, Real}

Return the imaginary part (also called vector part) of the quaternion `q`. The result is a
3-tuple of reals.

"""
Base.imag(q::AbstractQuaternion) = Tuple(q)[2:4]

Base.isreal(q::AbstractQuaternion) = (iszero(q.i) & iszero(q.j) & iszero(q.k))
Base.isinteger(q::AbstractQuaternion) = isreal(q) & isinteger(real(q))
Base.iszero(q::AbstractQuaternion) = isreal(q) & iszero(real(q))
Base.isone(q::AbstractQuaternion) = isreal(q) & isone(real(q))
function Base.isfinite(q::AbstractQuaternion{T}) where {T <: Real}
    return T <: Integer ? true : qmapfoldl(isfinite, &, q)
end
function Base.isnan(q::AbstractQuaternion{T}) where {T <: Real}
    return T <: Integer ? false : qmapfoldl(isnan, |, q)
end
function Base.isinf(q::AbstractQuaternion{T}) where {T <: Real}
    return T <: Integer ? false : qmapfoldl(isinf, |, q)
end

"""
    quaternion(s, i, j, k)
    quaternion(s, ijk = (0,0,0))
    quaternion(sijk)

Return a quaternion with scalar part `s` and vector part `(i,j,k)`. The vector part may
also be specified by a 3-tuple `ijk = (i,j,k)` or by a 3-vector. All components may also
be specified by the 4-tuple `sijk = (s, i, j, k)` or by a 4-vector.

The scalar and vector part are also called the *real* and *imaginary* part of the
quaternion.

"""
quaternion(s::Real) = Quaternion(s)
quaternion(sijk::Vararg{Real, 4}) = Quaternion(sijk)
quaternion(sijk::NTuple{4, Real}) = Quaternion(sijk)
quaternion(sijk::SVector{4, <: Real}) = quaternion(to_tuple(sijk))
quaternion(s::Real, ijk::NTuple{3, Real}) = quaternion(s, ijk...)
quaternion(s::Real, ijk::SVector{3, <: Real}) = quaternion(s, to_tuple(ijk))

"""
    normalize(q::AbstractQuaternion) -> u::UnitQuaternion

Return the unit quaternion `q/abs(q)`.

"""
normalize(q::UnitQuaternion) = q
normalize(q::AbstractQuaternion) = UnitQuaternion(Tuple(q))

Base.write(io::IO, q::AbstractQuaternion) = write(io, Tuple(q)...)
function Base.read(io::IO, ::Type{Quaternion{T}}) where {T <: Real}
    return Quaternion{T}(read(io, T), read(io, T), read(io, T), read(io, T))
end

# Byte order swaps: components are swapped individually.
Base.bswap(q::Quaternion) = Quaternion(qmap(bswap, q))
Base.bswap(q::UnitQuaternion) = _UnitQuaternion(qmap(bswap, q))

# TODO hash

# Quaternions form a vector space.
Base.:(+)(a::AbstractQuaternion, b::AbstractQuaternion) = Quaternion(qmap(+, a, b))
Base.:(-)(a::AbstractQuaternion, b::AbstractQuaternion) = Quaternion(qmap(-, a, b))
Base.:(*)(q::AbstractQuaternion, λ::Real) = λ * q
function Base.:(*)(λ::Real, q::AbstractQuaternion)
    l, s, i, j, k = promote(λ, Tuple(q)...)
    return Quaternion(l*s, l*i, l*j, l*k)
end
Base.:(-)(q::Quaternion) = Quaternion(qmap(-, q))
Base.:(-)(q::UnitQuaternion) = _UnitQuaternion(qmap(-, q))
Base.:(/)(q::AbstractQuaternion, λ::Real) = Quaternion(qmap(Base.Fix2(/, λ), q))


# TODO q + s::Real     like complex + real
# TODO q + v::Vector3  like complex + imag

# Conjugation, absolute value, norms, inverse, etc.
for (Q, f) in (AbstractQuaternion => Quaternion,
               UnitQuaternion => _UnitQuaternion)
    @eval Base.conj(q::$Q) = $f((q.s, -q.i, -q.j, -q.k))
end

Base.abs(q::UnitQuaternion{T}) where {T} = one(T)
Base.abs(q::AbstractQuaternion) = hypot(Tuple(q)...)
LinearAlgebra.norm(q::AbstractQuaternion) = abs(q)

Base.abs2(q::UnitQuaternion{T}) where {T} = one(T)
Base.abs2(q::AbstractQuaternion) = qmapfoldl(abs2, +, q)

Base.inv(q::UnitQuaternion) = conj(q)
function Base.inv(q::AbstractQuaternion)
    r = float(abs2(q))
    t = -r
    return Quaternion(q.s/r, q.i/t, q.j/t, q.k/t)
end

# TODO muladd
# TODO pow (^)

@inline qmap(f, q::AbstractQuaternion) = (f(q.s), f(q.i), f(q.j), f(q.k))
@inline qmap(f, a::AbstractQuaternion, b::AbstractQuaternion) =
    (f(a.s, b.s), f(a.i, b.i), f(a.j, b.j), f(a.k, b.k))

@inline qmapfoldl(f, op, q::AbstractQuaternion) = op(op(op(f(q.s), f(q.i)), f(q.j)), f(q.k))
@inline qmapfoldl(f, op, a::AbstractQuaternion, b::AbstractQuaternion) =
    op(op(op(f(a.s, b.s), f(a.i, b.i)), f(a.j, b.j)), f(a.k, b.k))

@inline qmapfoldr(f, op, q::AbstractQuaternion) = op(f(q.s), op(f(q.i), op(f(q.j), f(q.k))))
@inline qmapfoldr(f, op, a::AbstractQuaternion, b::AbstractQuaternion) =
    op(f(a.s, b.s), op(f(a.i, b.i), op(f(a.j, b.j), f(a.k, b.k))))

qvec(q::AbstractQuaternion) = q[2:4]
qdot(a::Vector3D, b::Vector3D) = mapreduce(*, +, a, b)
qcross(a::Vector3D, b::Vector3D) = (a[2]*b[3] - a[3]*b[2],
                                    a[3]*b[1] - a[1]*b[3],
                                    a[1]*b[2] - a[2]*b[1])

# Hamilton product (28 flops)
function qmul(a::NTuple{4, S}, b::NTuple{4, T}) where {S <: Real, T <: Real}
    R = promote_type(S, T)
    return qmul(convert(NTuple{4, R}, a), convert(NTuple{4, R}, b))
end
function qmul((a_s, a_i, a_j, a_k)::NTuple{4, T},
              (b_s, b_i, b_j, b_k)::NTuple{4, T}) where {T<:Real}
    return (a_s * b_s - a_i * b_i - a_j * b_j - a_k * b_k,
            a_s * b_i + a_i * b_s + a_j * b_k - a_k * b_j,
            a_s * b_j + a_j * b_s + a_k * b_i - a_i * b_k,
            a_s * b_k + a_k * b_s + a_i * b_j - a_j * b_i)
end

function Base.:(*)(a::AbstractQuaternion, b::AbstractQuaternion)
    return Quaternion(qmul(Tuple(a), Tuple(b)))
end
function Base.:(*)(a::UnitQuaternion, b::UnitQuaternion)
    return _UnitQuaternion(qmul(Tuple(a), Tuple(b)))
end

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
    @eval function Base.$f(a::AbstractQuaternion, b::AbstractQuaternion)
        return qmapfoldl($f, &, a, b)
    end
end
function isapprox(x::Number, y::Number;
                  atol::Real=0, rtol::Real=rtoldefault(x,y,atol),
                  nans::Bool=false, norm::Function=abs)
    x′, y′ = promote(x, y) # to avoid integer overflow
    x == y ||
        (isfinite(x) && isfinite(y) && norm(x-y) <= max(atol, rtol*max(norm(x′), norm(y′)))) ||
         (nans && isnan(x) && isnan(y))
end

function isapprox(x::Integer, y::Integer;
                  atol::Real=0, rtol::Real=rtoldefault(x,y,atol),
                  nans::Bool=false, norm::Function=abs)
    if norm === abs && atol < 1 && rtol == 0
        return x == y
    else
        # We need to take the difference `max` - `min` when comparing unsigned integers.
        _x, _y = x < y ? (x, y) : (y, x)
        return norm(_y - _x) <= max(atol, rtol*max(norm(_x), norm(_y)))
    end
end

const 𝐢 = _UnitQuaternion((0, 1, 0, 0))
const 𝐣 = _UnitQuaternion((0, 0, 1, 0))
const 𝐤 = _UnitQuaternion((0, 0, 0, 1))

Base.one(q::AbstractQuaternion) = one(typeof(q))
Base.one(::Type{<:AbstractQuaternion}) = one(AbstractQuaternion{Bool})
function Base.one(::Type{<:AbstractQuaternion{T}}) where {T <: Real}
    z = zero(T) # save memory for types such as big numbers
    return _UnitQuaternion(T, (one(T), z, z, z))
end

Base.zero(q::AbstractQuaternion) = zero(typeof(q))
Base.zero(::Type{<:AbstractQuaternion}) = zero(AbstractQuaternion{Bool})
function Base.zero(::Type{<:AbstractQuaternion{T}}) where {T <: Real}
    z = zero(T) # save memory for types such as big numbers
    return Quaternion((z, z, z, z))
end

#struct Im{S}; end

#const 𝐢 = Im{:i}()
#const 𝐣 = Im{:j}()
#const 𝐤 = Im{:k}()
#Base.(*)(::Im{:i}, ::Im{:i}) = -ONE
#Base.(*)(::Im{:j}, ::Im{:j}) = -ONE
#Base.(*)(::Im{:k}, ::Im{:k}) = -ONE
#
#Base.(*)(::Im{:i}, ::Im{:j}) = Im{:k}()
#Base.(*)(::Im{:j}, ::Im{:j}) = -ONE
#Base.(*)(::Im{:k}, ::Im{:k}) = -ONE
#
## i^2 = j^2 = k^2 = -1
## -> 1/i = -i
## -> 1/j = -j
## -> 1/k = -k
#Base.inv(i::Im{:i}) = -i
#Base.inv(j::Im{:j}) = -j
#Base.inv(k::Im{:k}) = -k
## i*j*k = -1
## -> i*j = -1/k = k
## -> j*k = i^{-1}*(i*j*k) = -i*(-1) = i
## -> j*i = j*i*j*k/k/j = j*(i*j*k)*(-k)*(-j) = -j*k*j = -i*j = -k
##

end
