# Very basic setup, useful for testing
using DFTK
using LazyArtifacts
using Interpolations
using LinearAlgebra
import DFTK: eval_psp_density_valence_fourier, eval_psp_projector_fourier, count_n_proj_radial, count_n_proj
import DFTK: cis2pi, ylm_real, enforce_real!, irfft, atomic_total_density, recip_vector_red_to_cart
import DFTK: build_projection_vectors_

a = 10.26;  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]];
Si = ElementPsp(:Si, psp=load_psp(artifact"pd_nc_sr_lda_standard_0.4.1_upf", "Si.upf"));
atoms     = [Si, Si];
positions = [ones(3)/8, -ones(3)/8];

model = model_LDA(lattice, atoms, positions);
basis = PlaneWaveBasis(model; Ecut=90, kgrid=[4, 4, 4]);
scfres = self_consistent_field(basis, tol=1e-8)

function build_interpolators_valence_charge(basis::PlaneWaveBasis{T},
                                            n_qnorm_interpolate::Int=3001) where {T}
    qnorm_max = maximum(norm.(G_vectors_cart(basis)))
    qnorm_interpolate = range(0, qnorm_max, n_qnorm_interpolate)

    map(basis.model.atom_groups) do atom_group
        psp = basis.model.atoms[first(atom_group)].psp
        f̃ = eval_psp_density_valence_fourier.(psp, qnorm_interpolate)
        scale(interpolate(f̃, BSpline(Cubic(Line(OnGrid())))), qnorm_interpolate)        
    end
end

function superposition_valence_charge(basis::PlaneWaveBasis{T}) where {T}
    itps = build_interpolators_valence_charge(basis)

    ρ = map(enumerate(G_vectors(basis))) do (iG, G)
        qnorm_cart = norm(recip_vector_red_to_cart(basis.model, G))
        ρ_G = sum(enumerate(basis.model.atom_groups); init=zero(Complex{T})) do (igroup, atom_group)
            form_factor = itps[igroup](norm(qnorm_cart))
            sum(atom_group) do iatom
                structure_factor = basis.structure_factors[iatom][iG]
                form_factor * structure_factor
            end
        end
        ρ_G / sqrt(basis.model.unit_cell_volume)
    end
    enforce_real!(basis, ρ)
    irfft(basis, ρ)
end

function build_interpolators_projectors(basis::PlaneWaveBasis{T},
                                        atom_groups::Vector{Vector{Int}},
                                        n_qnorm_interpolate::Int=3001) where {T}
    qnorm_max = maximum(maximum(norm.(Gplusk_vectors_cart(basis, kpt))) for kpt in basis.kpoints)
    qnorm_interpolate = range(0, qnorm_max, n_qnorm_interpolate)

    map(atom_groups) do atom_group
        psp = basis.model.atoms[first(atom_group)].psp
        map(0:psp.lmax) do l
            map(1:count_n_proj_radial(psp, l)) do iproj_l
                f̃ = eval_psp_projector_fourier.(psp, iproj_l, l, qnorm_interpolate)
                scale(interpolate(f̃, BSpline(Cubic(Line(OnGrid())))), qnorm_interpolate)
            end
        end
    end
end

function build_projection_vectors(basis::PlaneWaveBasis{T}, atom_groups::Vector{Vector{Int}}) where {T}
    itps = build_interpolators_projectors(basis, atom_groups)  # itps[group][l][n]
    psps = [basis.model.atoms[first(atom_group)].psp for atom_group in atom_groups]
    psp_positions = [basis.model.positions[atom_group] for atom_group in atom_groups]
    nproj = count_n_proj(psps, psp_positions)
    ngroup = length(atom_groups)
    lmax = maximum(psp.lmax for psp in psps)
    sqrt_Vuc = sqrt(basis.model.unit_cell_volume)

    proj_vectors = map(basis.kpoints) do kpt
        qs_cart = Gplusk_vectors_cart(basis, kpt)
        qnorms_cart = norm.(qs_cart)
        proj_vectors_k = zeros(Complex{T}, size(qs_cart, 1), nproj)

        angular = map(0:lmax) do l  # angular[l][m][q]
            map(-l:l) do m
                map(qs_cart) do q_cart      
                    im^l * ylm_real(l, m, q_cart)
                end
            end
        end

        radial = map(1:ngroup) do igroup  # radial[group][l][n][q]
            psp = psps[igroup]
            map(0:psp.lmax) do l
                map(1:count_n_proj_radial(psp, l)) do n
                    map(itps[igroup][l+1][n], qnorms_cart)
                end
            end
        end

        iproj = 1
        for igroup in 1:ngroup
            psp = psps[igroup]
            for iatom in atom_groups[igroup]
                for l in 0:psp.lmax
                    il = l + 1
                    for im in 1:(2l + 1)
                        for n in 1:count_n_proj_radial(psp, l)
                            aff = angular[il][im]  # Form-factor angular part
                            rff = radial[igroup][il][n]  # Form-factor radial part
                            sf = kpt.structure_factors[iatom]  # Structure factor
                            proj_vectors_k[:,iproj] .= sf .* aff .* rff ./ sqrt_Vuc
                            iproj += 1
                        end
                    end
                end
            end
        end
        proj_vectors_k
    end
    return proj_vectors
end

@benchmark ρ_old = atomic_total_density(basis, ValenceNumericalDensity())
@benchmark ρ_new = superposition_valence_charge(basis)

extrema(abs.(ρ_old .- ρ_new))

VSCodeServer.@profview ρ_old = atomic_total_density(basis, ValenceNumericalDensity())
VSCodeServer.@profview ρ_new = superposition_valence_charge(basis)

function build_proj_vectors_old(basis::PlaneWaveBasis)
    psps = [basis.model.atoms[first(atom_group)].psp for atom_group in basis.model.atom_groups];
    psp_positions = [basis.model.positions[atom_group] for atom_group in basis.model.atom_groups];
    [build_projection_vectors_(basis, kpt, psps, psp_positions) for kpt in basis.kpoints];
end

@benchmark P_old = build_proj_vectors_old(basis)
@benchmark P_new = build_projection_vectors(basis, basis.model.atom_groups)

for (Pi_old, Pi_new) in zip(P_old, P_new)
    real_errs = extrema(abs.(real.(Pi_old .- Pi_new)))
    imag_errs = extrema(abs.(imag.(Pi_old .- Pi_new)))
    println("Real: $real_errs    Imag: $imag_errs")
end

VSCodeServer.@profview build_projection_vectors_(basis, basis.kpoints[1], psps, psp_positions);
VSCodeServer.@profview build_projection_vectors(basis, basis.model.atom_groups);