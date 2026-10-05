# QuaternionNumbers [![Build Status](https://github.com/emmt/QuaternionNumbers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/QuaternionNumbers.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/emmt/QuaternionNumbers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/QuaternionNumbers.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Package `QuaternionNumbers` provides quaternions whose instances are `Number`s.

The package exports 3 types (`AbstractQuaternion{T} <: Number`, `Quaternion{T} <:
AbstractQuaternion{T}` and `Versor{T} <: AbstractQuaternion{T}`, with `T <: Real` the type
of the components of the quaternion) and 2 functions:

* `quaternion(args...)::Quaternion` builds a quaternion from arguments `args...`.

* `normalize(q::AbstractQuaternion)::Versor` yields the unit quaternion `q/abs(q)`.

The scalar and vector parts of a quaternion `q` are given by `real(q)` and `imag(q)`.
