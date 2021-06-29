# Computation of error estimate and corrections for the forces for the linear
# silicon system, in the form Ax=b
#
# Very basic setup, useful for testing
using DFTK
import DFTK: apply_K, apply_Ω, solve_ΩplusK, compute_projected_gradient
import DFTK: proj_tangent, proj_tangent!, proj_tangent_kpt
using HDF5
using PyPlot

include("aposteriori_forces.jl")
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8 + [0.42, 0.35, 0.24] ./ 20, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 200   # kinetic energy cutoff in Hartree
tol = 1e-10
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)

filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
Nk = length(basis_ref.kpoints)
T = eltype(basis_ref)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

#  scfres_ref = self_consistent_field(basis_ref, tol=tol,
#                                     determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
#                                     is_converged=DFTK.ScfConvergenceDensity(tol))

## reference values
φ_ref = similar(scfres_ref.ψ)
for ik = 1:Nk
    φ_ref[ik] = scfres_ref.ψ[ik][:,1:N]
end
f_ref = compute_forces(scfres_ref)

## min and max Ecuts for the two grid solution
Ecut_min = 5
Ecut_max = 80

Ecut_list = Ecut_min:5:Ecut_max
K = length(Ecut_list)
diff_list = zeros((K,K))
diff_list_res = zeros((K,K))
diff_list_schur = zeros((K,K))

i = 0
j = 0

for Ecut_g in Ecut_list

    println("---------------------------")
    println("Ecut grossier = $(Ecut_g)")
    global i,j
    i += 1
    j = i
    basis_g = PlaneWaveBasis(model, Ecut_g; kgrid=kgrid)

    ## solve eigenvalue system
    scfres_g = self_consistent_field(basis_g, tol=tol,
                                     determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                     is_converged=DFTK.ScfConvergenceDensity(tol))
    ham_g = scfres_g.ham
    ρ_g = scfres_g.ρ

    ## quantities
    φ = similar(scfres_g.ψ)
    for ik = 1:Nk
        φ[ik] = scfres_g.ψ[ik][:,1:N]
    end
    f_g = compute_forces(scfres_g)

    for Ecut_f in Ecut_g:5:Ecut_max

        println("Ecut fin = $(Ecut_f)")
        # fine grid
        basis_f = PlaneWaveBasis(model, Ecut_f; kgrid=kgrid)

        # compute residual and keep only LF
        φr = DFTK.transfer_blochwave(φ, basis_g, basis_f)
        res = compute_projected_gradient(basis_f, φr, occupation)
        resLF = DFTK.transfer_blochwave(res, basis_f, basis_g)

        # compute hamiltonian
        ρr = compute_density(basis_f, φr, occupation)
        _, ham_f = energy_hamiltonian(basis_f, φr, occupation; ρ=ρr)

        ## prepare P
        kpt = basis_f.kpoints[1]
        P = [PreconditionerTPA(basis_f, kpt) for kpt in basis_f.kpoints]
        for ik = 1:length(P)
            DFTK.precondprep!(P[ik], φr[ik])
        end

        ## Rayleigh coefficients
        Λ = map(enumerate(φr)) do (ik, ψk)
            Hk = ham_f.blocks[ik]
            Hψk = Hk * ψk
            ψk'Hψk
        end

        resHF = res - DFTK.transfer_blochwave(resLF, basis_g, basis_f)
        resHF = apply_metric(φr, P, resHF, apply_inv_T)
        ΩpKres = apply_Ω(resHF, φr, ham_f, Λ) .+ apply_K(basis_f, resHF, φr, ρr, occupation)
        ΩpKresLF = DFTK.transfer_blochwave(ΩpKres, basis_f, basis_g)
        rhs = resLF - ΩpKresLF
        eLF = solve_ΩplusK(basis_g, φ, rhs, occupation)
        e = DFTK.transfer_blochwave(eLF, basis_g, basis_f)

        # Apply M^+-1/2
        Me = apply_metric(φr, P, e, apply_sqrt_M)
        Mres = apply_metric(φr, P, res, apply_inv_sqrt_M)
        Mschur = [Mres[1] + Me[1]]

        # approximate forces f-f*
        f_res = compute_forces_estimate(basis_f, Mres, φr, P, occupation)
        f_schur = compute_forces_estimate(basis_f, Mschur, φr, P, occupation)

        diff_list[i,j] = abs(f_g[1][2][1]-f_ref[1][2][1])
        diff_list_res[i,j] = abs(f_g[1][2][1]-f_res[1][2][1]-f_ref[1][2][1])
        diff_list_schur[i,j] = abs(f_g[1][2][1]-f_schur[1][2][1]-f_ref[1][2][1])
        j+=1
    end
end

h5open("3_grids_forces.h5", "w") do file
    println("writing h5 file")
    file["Ecut_ref"] = Ecut_ref
    file["Ecut_list"] = collect(Ecut_list)
    file["diff_list"] = diff_list
    file["diff_list_res"] = diff_list_res
    file["diff_list_schur"] = diff_list_schur
end

