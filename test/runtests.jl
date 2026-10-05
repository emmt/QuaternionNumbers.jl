module TestingQuaternionNumbers

using QuaternionNumbers
using Test
using Aqua

@testset "Quaternion numbers" begin

    let i = QuaternionNumbers.𝐢, j = QuaternionNumbers.𝐣, k = QuaternionNumbers.𝐤
        @test isone(@inferred(abs(i)))
        @test isone(@inferred(abs(j)))
        @test isone(@inferred(abs(k)))
        @test @inferred(1*i) == i
        @test @inferred(i*1) == i
        @test @inferred(-1*i) == -i
        @test @inferred(i*-1) == -i
        ii = @inferred(i*i)
        @test isreal(ii)
        @test @inferred(real(ii)) === -1
        @test ii === @inferred(-one(i))
        jj = @inferred(j*j)
        @test isreal(jj)
        @test @inferred(real(jj)) === -1
        @test jj === @inferred(-one(i))
        kk = @inferred(k*k)
        @test isreal(kk)
        @test @inferred(real(kk)) === -1
        @test kk === @inferred(-one(i))
        ijk = @inferred(i*j*k)
        @test isreal(ijk)
        @test @inferred(real(ijk)) === -1
        @test ijk === @inferred(-one(i))
        @test @inferred(i*j) == k
        @test @inferred(j*i) === -k
        @test @inferred(j*k) == i
        @test @inferred(k*j) === -i
        @test @inferred(k*i) == j
        @test @inferred(i*k) === -j
    end

    @test @inferred(big(AbstractQuaternion{Bool}))    === AbstractQuaternion{BigInt}
    @test @inferred(big(AbstractQuaternion{UInt}))    === AbstractQuaternion{BigInt}
    @test @inferred(big(AbstractQuaternion{Float16})) === AbstractQuaternion{BigFloat}

    @test @inferred(big(Versor{Bool}))    === Versor{BigInt}
    @test @inferred(big(Versor{Int16}))   === Versor{BigInt}
    @test @inferred(big(Versor{Float32})) === Versor{BigFloat}

    @test @inferred(big(Quaternion{Bool}))    === Quaternion{BigInt}
    @test @inferred(big(Quaternion{UInt64}))  === Quaternion{BigInt}
    @test @inferred(big(Quaternion{Float64})) === Quaternion{BigFloat}

    @testset "Code quality (Aqua)" begin
        Aqua.test_all(QuaternionNumbers)
    end
end

end # module
