using QuaternionNumbers
using Test
using Aqua

@testset "QuaternionNumbers.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(QuaternionNumbers)
    end
    # Write your tests here.
end
