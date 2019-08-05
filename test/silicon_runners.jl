using Test
using DFTK
using Libxc: Functional

include("silicon_testcases.jl")

# TODO There is a lot of code duplication in this file ... once we have the ABINIT reference
#      stuff in place, this should be refactored.

function run_silicon_noXC(;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using ABINIT with the same kpoints as silicon_testcases.jl
    # and Ecut = 25
    ref_noXC = [
        [0.185624211511768, 0.645877543801093, 0.645877543804049, 0.645877543807010,
         0.710639433524012, 0.710639433527887, 0.710639433531763, 0.747770226622483,
         0.846692796204658, 0.903426057105463],
        [0.239766326671308, 0.428509722981838, 0.603258008732289, 0.603258008735511,
         0.677812317259065, 0.740270198505120, 0.740270198508514, 0.855980714202823,
         0.904920991452396, 0.904920991455659],
        [0.259399404104112, 0.456356401369978, 0.541006959363826, 0.541006959368758,
         0.620449688893274, 0.658935639559930, 0.894581352229539, 0.894581352233318,
         0.982503349423807, 0.982875170859166],
        [0.316557263535848, 0.385639894339023, 0.446179301437491, 0.554693942841539,
         0.649929646792239, 0.820125934482662, 0.856479348217456, 0.880349468106374,
         0.884889516616732, 0.892844865916440],
    ]
    n_bands = length(ref_noXC[1])

    basis = PlaneWaveBasis(lattice, grid_size * ones(3), Ecut, kpoints, kweights, ksymops)
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    n_electrons = length(positions) * n_elec_valence(Si)

    # Construct a Hamiltonian (Kinetic + local psp + nonlocal psp + Hartree)
    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis))

    ρ = guess_gaussian_sad(basis, Si => positions)
    prec = PreconditionerKinetic(ham, α=0.1)
    scfres = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                   tol=scf_tol)

    for ik in 1:length(kpoints)
        println(ik, "  ", abs.(ref_noXC[ik] - scfres.orben[ik]))
    end
    for ik in 1:length(kpoints)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_noXC[ik] - scfres.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end
end

function run_silicon_lda(T ;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # These values were computed using ABINIT with the same kpoints as silicon_testcases.jl
    # and Ecut = 25
    ref_lda = [
        [-0.178566465714968, 0.261882541175914, 0.261882541178847, 0.261882541181782,
          0.354070367072414, 0.354070367076363, 0.354070367080310, 0.376871160884678],
        [-0.127794342370963, 0.064395861472044, 0.224958824747686, 0.224958824750934,
          0.321313617512188, 0.388442495007398, 0.388442495010722, 0.542078732298094],
        [-0.108449612789883, 0.077125812982728, 0.172380374761464, 0.172380374766260,
          0.283802499666810, 0.329872296009131, 0.525606867582028, 0.525606867585921],
        [-0.058089253154566, 0.012364292440522, 0.097350168867990, 0.183765652148129,
          0.314593174568090, 0.470869435132365, 0.496966579772700, 0.517009645871194],
    ]
    ref_etot = -7.911817522631488
    n_bands = length(ref_lda[1])

    grid_size = grid_size * ones(3)
    basis = PlaneWaveBasis(Array{T}(lattice), grid_size, Ecut, kpoints, kweights, ksymops)
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    n_electrons = length(positions) * n_elec_valence(Si)

    # Construct the Hamiltonian
    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis),
                      pot_xc=PotXc(basis, :lda_x, :lda_c_vwn))

    # Construct guess and run the SCF
    ρ = guess_gaussian_sad(basis, Si => positions)
    prec = PreconditionerKinetic(ham, α=0.1)
    scfres = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                   tol=scf_tol)

    for ik in 1:length(kpoints)
        @test eltype(scfres.orben[ik]) == T
        @test eltype(scfres.Psi[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_lda[ik] - scfres.orben[ik]))
    end
    for ik in 1:length(kpoints)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_lda[ik] - scfres.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end

    energies = scfres.energies
    energies[:Ewald] = energy_nuclear_ewald(basis.lattice, Si => positions)
    energies[:PspCorrection] = energy_nuclear_psp_correction(basis.lattice, Si => positions)
    @test sum(values(energies)) ≈ ref_etot atol=test_tol
end


function run_silicon_pbe(T ;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # These values were computed using ABINIT with the same kpoints as silicon_testcases.jl
    # and Ecut = 25
    ref_pbe = [
        [-0.182613872647344, 0.259343145910962, 0.259343145913851, 0.259343145916741,
          0.350081372804469, 0.350081372808377, 0.350081372812286, 0.379795826432543,
          0.535074885965206, 0.539275063324970],
        [-0.131467986580421, 0.060790353181110, 0.222093614848986, 0.222093614852180,
          0.319157964354942, 0.383670764757857, 0.383670764761167, 0.537726468987132,
          0.547524586026887, 0.547524586030377],
        [-0.112051979072326, 0.074925981269773, 0.168690502706886, 0.168690502711648,
          0.278391665927345, 0.323679635165690, 0.523474718695926, 0.523474718699762,
          0.615008723926775, 0.618278015433293],
        [-0.061062886690018, 0.009486666207978, 0.092232334679323, 0.180288242428985,
          0.309085542639357, 0.465861199189178, 0.493919656233940, 0.515222260730036,
          0.527429170390239, 0.535275889950469],
    ]
    ref_etot = -7.838446229475437
    n_bands = length(ref_pbe[1])

    grid_size = grid_size * ones(3)
    basis = PlaneWaveBasis(Array{T}(lattice), grid_size, Ecut, kpoints, kweights, ksymops)
    Si = Species(atnum, psp=load_psp("si-pbe-q4.hgh"))
    n_electrons = length(positions) * n_elec_valence(Si)

    # Construct the Hamiltonian
    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis),
                      pot_xc=PotXc(basis, :gga_x_pbe, :gga_c_pbe))

    # Construct guess and run the SCF
    ρ = guess_gaussian_sad(basis, Si => positions)
    prec = PreconditionerKinetic(ham, α=0.1)
    scfres = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                   tol=scf_tol)

    for ik in 1:length(kpoints)
        @test eltype(scfres.orben[ik]) == T
        @test eltype(scfres.Psi[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_pbe[ik] - scfres.orben[ik]))
    end
    for ik in 1:length(kpoints)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_pbe[ik] - scfres.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end

    energies = scfres.energies
    energies[:Ewald] = energy_nuclear_ewald(basis.lattice, Si => positions)
    energies[:PspCorrection] = energy_nuclear_psp_correction(basis.lattice, Si => positions)
    @test sum(values(energies)) ≈ ref_etot atol=test_tol
end
