using Test
using DFTK: core_density_superposition
using Downloads
using LinearAlgebra

base_url = "https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary/main/pseudos/"
pseudo_urls = Dict(
    # With NLCC
    :Si => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Si.upf"),
    :Fe => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Fe.upf"),
    # Without NLCC
    :Li => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Li.upf"),
    :Mg => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Mg.upf")
)

@testset "Core charge density is positive" begin
    lattice = 5 * I(3)
    positions = [zeros(3)]
    for (element, element_url) in pseudo_urls
        psp = load_psp(Downloads.download(element_url, joinpath(tempdir(), "psp.upf")))
        atoms = [ElementPsp(element, psp=psp)]
        model = model_LDA(lattice, atoms, positions)
        basis = PlaneWaveBasis(model; Ecut=12, kgrid=[4, 4, 4])
        ρ_core = core_density_superposition(basis)
        ρ_core_neg = abs(sum(ρ_core[ρ_core .< 0]))
        @test ρ_core_neg * model.unit_cell_volume / prod(basis.fft_size) < 1e-6
    end
end
