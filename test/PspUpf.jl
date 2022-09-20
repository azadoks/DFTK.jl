using Test
using DFTK: load_psp, eval_psp_projector_fourier, eval_psp_local_fourier
using DFTK: eval_psp_projector_real, eval_psp_local_real
using SpecialFunctions: besselj
using QuadGK

@testset "Check upf parsing" begin
    pspfile = "Si.nc.z_4.oncvpsp3.dojo.v4-std.upf"
    psp = load_psp("upf/pbe/" * pspfile)

    @test psp.lmax == 2
    @test psp.Zion == 4
    @test length(psp.rgrid) == 1510
    @test length(psp.lpot) == 1510
    for m in psp.h
        @test size(m) == (2, 2)
    end

    @test psp.projs[1][1][1] ≈ -5.6328824383E-09
end

@testset "Potentials are consistent in real and Fourier space" begin
    reg_param = 0.001  # divergent integral, needs regularization
    function integrand(psp, q, r)
        j0qr = sin(2π * q * r)/(2π * q)
        4π * j0qr * (r * eval_psp_local_real(psp, r) + psp.Zion)
    end

    for pspfile in ["Si.nc.z_4.oncvpsp3.dojo.v4-std.upf"]
        psp = load_psp("upf/pbe/" * pspfile)
        for q in [0.01, 0.2, 1, 1.3]
            reference = quadgk(r -> integrand(psp, q, r), psp.rgrid[1], psp.rgrid[end])[1]
            @test reference ≈ eval_psp_local_fourier(psp, q) rtol=.1 atol = .1
        end
    end
end

@testset "Projectors are consistent in real and Fourier space" begin
    # The spherical bessel function of the first kind in terms of ordinary bessels:
    function j(n, x::T) where {T}
        x == 0 ? zero(T) : sqrt(π/2x) * besselj(n+1/2, x)
    end

    # The integrand for performing the spherical Hankel transfrom,
    # i.e. compute the radial part of the projector in Fourier space
    function integrand(psp, i, l, q, x)
        4π * x^2 * eval_psp_projector_real(psp, i, l, x) * j(l, q*x)
    end

    for pspfile in ["Si.nc.z_4.oncvpsp3.dojo.v4-std.upf"]
        psp = load_psp("upf/pbe/" * pspfile)
        for (l, i) in [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                       (2, 1), (2, 2), (3, 1)]
            l > psp.lmax - 1 && continue  # Overshooting available AM
            for q in [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]
                reference = quadgk(r -> integrand(psp, i, l, q, r), psp.rgrid[1], psp.rgrid[end])[1]
                @test reference ≈ eval_psp_projector_fourier(psp, i, l, q) atol=5e-15 rtol=1e-8
            end
        end
    end
end