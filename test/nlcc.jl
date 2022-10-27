using Test
using DFTK: core_density_superposition
using Downloads
using LinearAlgebra

base_url = "https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary/main/pseudos/"
pseudo_urls = Dict(
    # With NLCC
    :Si => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Si.upf"),
    :Fe => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Fe.upf"),
    :Ir => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Ir.upf"),
    # Without NLCC
    :Li => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Li.upf"),
    :Mg => joinpath(base_url, "pd_nc_sr_lda_standard_04_upf/Mg.upf")
)
pseudos = Dict(
    key => load_psp(Downloads.download(value, joinpath(tempdir(), "$(string(key)).upf")))
    for (key, value) in pseudo_urls
)

@testset "Core charge density is positive" begin
    lattice = 5 * I(3)
    positions = [zeros(3)]
    for (element, psp) in pseudos
        atoms = [ElementPsp(element, psp=psp)]
        model = model_LDA(lattice, atoms, positions)
        basis = PlaneWaveBasis(model; Ecut=24, kgrid=[2, 2, 2])
        ρ_core = core_density_superposition(basis)
        ρ_core_neg = abs(sum(ρ_core[ρ_core .< 0]))
        @test ρ_core_neg * model.unit_cell_volume / prod(basis.fft_size) < 1e-6
    end
end
