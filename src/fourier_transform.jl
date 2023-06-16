function atomic_density_superposition(basis::PlaneWaveBasis{T}, quantity::AtomicDensity; coefficients=ones(T, length(basis.model.atoms))) where {T}
    model = basis.model
    atom_groups = model.atom_groups
    atom_group_elements = [model.atoms[first(group)] for group in atom_groups]
    form_factor_functions = [form_factor_function(basis, element, quantity) for element in atom_group_elements]

    ρ = map(enumerate(G_vectors(basis))) do (iG, G)
        qnorm_cart = norm(recip_vector_red_to_cart(basis.model, G))
        ρ_G = sum(enumerate(atom_groups); init=zero(Complex{T})) do (igroup, group)
            form_factor = form_factor_functions[igroup](qnorm_cart)
            sum(group) do iatom
                structure_factor = basis.structure_factors[iatom][iG]
                coefficients[iatom] * form_factor * structure_factor
            end
        end
        ρ_G / sqrt(model.unit_cell_volume)
    end
    enforce_real!(basis, ρ)
    irfft(basis, ρ)
end

function atomic_density_superposition(basis::PlaneWaveBasis{T}, quantity::DensityFlag; coefficients=ones(T, length(basis.atoms))) where {T}
    qs_cart = G_vectors_cart(basis)
    qnorms_cart = G_norms_cart(basis)  # TODO: to implement

    ρ_atom = zeros_like(qnorms_cart)

    sum(zip(basis.atoms, coefficients)) do (atom, coefficient)
        atomic_density = get_quantity(atom, quantity)
        ρ_atom .= coefficient
        ρ_atom .*= atomic_density.(qnorms_cart)  # Form factors
        ρ_atom .*= cis2pi.(-sum(qs_cart .* atom.position, dims=1))  # Structure factors
    end
end

function atomic_state_bloch_transform(basis::PlaneWaveBasis{T}, atom::Element, quantity::ProjectorFlag)
    r = atom.position
    map(basis.kpoints) do kpt
        qs_frac = Gplusk_vectors(basis, kpt)
        qs_cart = Gplusk_vectors_cart(basis, kpt)
        qnorms_cart = Gplusk_norms_cart(qs_cart)  # TODO: to implement
        bloch_orbitals_k = zeros(Complex{T}, length(qnorms_cart), n_angulars(atom))

        i_orbital = 1
        for l in angular_momenta(atom), m in -l:l, n in 1:n_radials(atom, quantity, l)
            atomic_orbital = get_quantity(atom, quantity, l, n)
            bloch_orbitals_k[:,i_orbital] .= (
                (-im)^l .* ylm_real.(l, m, qs_cart) .*  # Angular part
                atomic_orbital.(qnorms_cart) .*  # Radial part
                [cis2pi(-dot(q, r)) for q in qs_frac] ./ # Structure factor
                sqrt(unit_cel_volume)  # Normalization
            )
        end
        bloch_orbitals_k
    end
end










abstract type AtomicQuantity end

struct AtomicDensity <: AtomicQuantity end
struct GaussianValenceChargeDensity <: AtomicDensity end
struct PseudoValenceChargeDensity <: AtomicDensity end
struct AutoValenceChargeDensity <: AtomicDensity end
struct PseudoCoreChargeDensity <: AtomicDensity end

struct AtomicOrbital <: AtomicQuantity end
struct PseudoBetaProjector <: AtomicQuantity end
struct PseudoChiProjector <: AtomicQuanity end

n_orbital_radials(::PseudoBetaProjector, psp::PseudoPotentialIO.AbstractPsP, l) = PseudoPotentialIO.n_projector_radials(psp, l)
n_orbital_radials(::PseudoChiProjector, psp::PseudoPotentialIO.AbstractPsP, l) = PseudoPotentialIO.n_chi_radials(psp, l)
n_orbital_radials(quantity::AtomicQuantity, psp::PseudoPotentialIO.AbstractPsP) = sum(l -> PseudoPotentialIO.n_orbital_radials(quantity, psp, l), PseudoPotentialIO.angular_momenta(psp))

n_orbital_angulars(::PseudoBetaProjector, psp::PseudoPotentialIO.AbstractPsP, l) = PseudoPotentialIO.n_projector_angulars(psp, l)
n_orbital_angulars(::PseudoChiProjector, psp::PseudoPotentialIO.AbstractPsP, l) = PseudoPotentialIO.n_chi_angulars(psp, l)
n_orbital_angulars(quantity::AtomicQuantity, psp::PseudoPotentialIO.AbstractPsP) = sum(l -> n_orbital_angulars(quantity, psp, l), PseudoPotentialIO.angular_momenta(psp))
function n_orbital_angulars(::AtomicQuantity, psps, psp_positions)
    sum(n_orbital_angulars(quantity, psp) * length(positions)
        for (psp, positions) in zip(psps, psp_positions))
end

orbital_radial_indices(::PseudoBetaProjector, psp::PseudoPotentialIO.AbstractPsP, l) = PseudoPotentialIO.projector_radial_indices(psp, l)
orbital_radial_indices(::PseudoChiProjector, psp::PseudoPotentialIO.AbstractPsP, l) = PseudoPotentialIO.chi_radial_indices(psp, l)

function atomic_density_superposition(basis::PlaneWaveBasis{T}, quantity::AtomicDensity; coefficients=ones(T, length(basis.model.atoms))) where {T}
    model = basis.model
    atom_groups = model.atom_groups
    atom_group_elements = [model.atoms[first(group)] for group in atom_groups]
    ff_functions = map(el -> form_factor_function(basis, quantity, el), atom_group_elements)

    ρ = map(enumerate(G_vectors(basis))) do (iG, G)
        qnorm_cart = norm(recip_vector_red_to_cart(basis.model, G))
        ρ_G = sum(enumerate(atom_groups); init=zero(Complex{T})) do (igroup, group)
            form_factor = ff_functions[igroup](qnorm_cart)
            sum(group) do iatom
                structure_factor = basis.structure_factors[iatom][iG]
                coefficients[iatom] * form_factor * structure_factor
            end
        end
        ρ_G / sqrt(model.unit_cell_volume)
    end
    enforce_real!(basis, ρ)
    irfft(basis, ρ)
end

function form_factor_function(::PlaneWaveBasis{T}, ::GaussianValenceChargeDensity, el::Element) where {T}
    qnorm -> gaussian_valence_charge_density_fourier(el, qnorm)
end

function form_factor_function(basis::PlaneWaveBasis{T}, ::PseudoValenceChargeDensity, el::ElementPsp; n_qnorm_interpolate::Int=3001) where {T}
    qnorm_max = maximum(norm.(G_vectors_cart(basis)))
    qnorm_interpolate = range(0, qnorm_max, n_qnorm_interpolate)
    f̃ = eval_psp_density_valence_fourier.(el.psp, qnorm_interpolate)
    scale(interpolate(f̃, BSpline(Cubic(Line(OnGrid())))), qnorm_interpolate)
end

function form_factor_function(basis::PlaneWaveBasis{T}, ::AutoValenceChargeDensity, el::Element) where {T}
    quantity = has_valence_density(el) ? PseudoValenceChargeDensity() : GaussianValenceChargeDensity()
    return form_factor_function(basis, quantity, el)
end

function form_factor_function(basis::PlaneWaveBasis{T}, ::PseudoCoreChargeDensity, el::Element; n_qnorm_interpolate::Int=3001) where {T}
    if has_density_core(el)
        qnorm_max = maximum(norm.(G_vectors_cart(basis)))
        qnorm_interpolate = range(0, qnorm_max, n_qnorm_interpolate)
        f̃ = eval_psp_density_core_fourier.(el.psp, qnorm_interpolate)
        return scale(interpolate(f̃, BSpline(Cubic(Line(OnGrid())))), qnorm_interpolate)
    else
        return qnorm -> zero(qnorm)
    end
end

@doc raw"""
Get the lengthscale of the valence density for an atom with `n_elec_core` core
and `n_elec_valence` valence electrons.
"""
function atom_decay_length(n_elec_core, n_elec_valence)
    # Adapted from ABINIT/src/32_util/m_atomdata.F90,
    # from which also the data has been taken.

    n_elec_valence = round(Int, n_elec_valence)
    if n_elec_valence == 0
        return 0.0
    end

    data = if n_elec_core < 0.5
        # Bare ions: Adjusted on 1H and 2He only
        [0.6, 0.4, 0.3, 0.25, 0.2]
    elseif n_elec_core < 2.5
        # 1s2 core: Adjusted on 3Li, 6C, 7N, and 8O
        [1.8, 1.4, 1.0, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
    elseif n_elec_core < 10.5
        # Ne core (1s2 2s2 2p6): Adjusted on 11na, 13al, 14si and 17cl
        [2.0, 1.6, 1.25, 1.1, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.6]
    elseif n_elec_core < 12.5
        # Mg core (1s2 2s2 2p6 3s2): Adjusted on 19k, and on n_elec_core==10
        [1.9, 1.5, 1.15, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.5]
    elseif n_elec_core < 18.5
        # Ar core (Ne + 3s2 3p6): Adjusted on 20ca, 25mn and 30zn
        [2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.65, 0.6]
    elseif n_elec_core < 28.5
        # Full 3rd shell core (Ar + 3d10): Adjusted on 31ga, 34se and 38sr
        [1.5, 1.25, 1.15, 1.05, 1.00, 0.95, 0.95, 0.9, 0.9, 0.85, 0.85, 0.80,
            0.8, 0.75, 0.7]
    elseif n_elec_core < 36.5
        # Krypton core (Ar + 3d10 4s2 4p6): Adjusted on 39y, 42mo and 48cd
        [2.0, 2.00, 1.60, 1.40, 1.25, 1.10, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.7]
    else
        # For the remaining elements, consider a function of n_elec_valence only
        [2.0, 2.00, 1.55, 1.25, 1.15, 1.10, 1.05, 1.0, 0.95, 0.9, 0.85, 0.85, 0.8]
    end
    data[min(n_elec_valence, length(data))]
end
atom_decay_length(sp::Element) = atom_decay_length(n_elec_core(sp), n_elec_valence(sp))

# TODO: rethink the data structure for the projectors / bloch orbitals
# TODO: there must be a better way to lay it out so that the construction is simpler
# TODO: and the evaluation of the NL potential is not too complicated

# Construct Bloch orbitals on the plane-wave basis from an atomic orbital at angular
# momentum `l` and magnetic quantum number `m`
# Note: This is not as efficient as it could be; the orbital is re-evaluated for each
# `m` even though it does not depend on it.
function atomic_orbital_to_bloch(basis::PlaneWaveBasis{T}, orbital_fourier, l::Int, m::Int,
                                 iatom::Int, kpoint::Kpoint) where {T}
    qs = Gplusk_vectors_cart(basis, kpoint)
    q_norms = norm.(qs)
    orbital_bloch = to_device(basis.architecture, zeros(Complex{T}, length(q_norms)))
    orbital_bloch .= (im^l .* ylm_real.(l, m, q_norms) .* orbital_fourier.(q_norms) .*
                      kpoint.structure_factors[iatom] ./ sqrt(basis.model.unit_cell_volume))
    return orbital_bloch
end

function build_orbital_fouriers(basis::PlaneWaveBasis{T}, quantity::AtomicOrbital, atom_groups, n_interpolate=3001)
    qnorm_max = maximum(maximum(norm.(Gplusk_vectors_cart(basis, kpt))) for kpt in basis.kpoints)
    qnorm_interpolate = range(0, qnorm_max, n_interpolate)

    map(atom_groups) do atom_group
        psp = basis.model.atoms[first(atom_group)].psp
        map(angular_momenta(psp)) do l
            map(orbital_radial_indices(quantity, psp, l)) do n
                ϕ̃ = orbital_fourier(quantity, psp, l, n)
                scale(interpolate(ϕ̃.(qnorm_interpolate), BSpline(Cubic(Line(OnGrid())))), qnorm_interpolate)
            end
        end
    end
end

function atomic_orbitals_to_bloch(basis::PlaneWaveBasis{T}, quantity::AtomicOrbital, atom_groups, orbital_fouriers)
    psps = [basis.model.atoms[first(atom_group)].psp for atom_group in atom_groups]
    psp_positions = [basis.model.positions[atom_group] for atom_group in atom_groups]
    nproj = n_orbital_angulars(quantity, psps, psp_positions)
    lmax = maximum(psp.lmax for psp in psps)

    map(basis.kpoints) do kpt
        bloch_orbitals_k = to_device(basis.architecture, zeros(Complex{T}, size(qs_cart, 1), nproj))
        iorb = 1
        for igroup in eachindex(atom_groups)
            psp = psps[igroup]
            for iatom in atom_groups[igroup]
                for l in 0:lmax
                    for m in -l:l
                        for n in 1:PseudoPotentialIO.n_orbital_radials(quantity, psp, l)
                            bloch_orbitals_k[:,iorb] .= atomic_orbital_to_bloch(
                                basis, orbital_fouriers[igroup][l+1][n], l, m, iatom, kpt
                            )
                            iorb += 1
                        end
                    end
                end
            end
        end
        bloch_orbitals_k
    end
end

# Construct Bloch orbitals on the plane-wave basis from all atomic orbitals at all k-points
function build_projection_vectors(basis::PlaneWaveBasis{T}, atom_groups, projector_fouriers) where {T}
    psps = [basis.model.atoms[first(atom_group)].psp for atom_group in atom_groups]
    psp_positions = [basis.model.positions[atom_group] for atom_group in atom_groups]
    nproj = PseudoPotentialIO.n_projector_angulars(psps, psp_positions)
    lmax = maximum(psp.lmax for psp in psps)
    sqrt_Ω = sqrt(basis.model.unit_cell_volume)

    proj_vectors = map(basis.kpoints) do kpt
        qs_cart = Gplusk_vectors_cart(basis, kpt)
        qnorms_cart = norm.(qs_cart)
        proj_vectors_k = to_device(basis.architecture, zeros(Complex{T}, size(qs_cart, 1), nproj))

        angular = map(0:lmax) do l  # angular[l][m][q]
            map(-l:l) do m
                map(qs_cart) do q_cart
                    im^l * ylm_real(l, m, q_cart)
                end
            end
        end

        radial = map(eachindex(atom_groups)) do igroup  # radial[group][l][n][q]
            psp = psps[igroup]
            map(0:lmax) do l
                map(PseudoPotentialIO.projector_radial_indices(psp, l)) do n
                    map(projector_fouriers[igroup][l+1][n], qnorms_cart)
                end
            end
        end

        iproj = 1
        for igroup in eachindex(atom_groups)
            psp = psps[igroup]
            for iatom in atom_groups[igroup]
                for l in 0:lmax
                    il = l + 1
                    for im in 1:(2l + 1)
                        for n in PseudoPotentialIO.projector_radial_indices(psp, l)
                            aff = angular[il][im]  # Form-factor angular part
                            rff = radial[igroup][il][n]  # Form-factor radial part
                            sf = kpt.structure_factors[iatom]  # Structure factor
                            proj_vectors_k[:,iproj] .= sf .* aff .* rff ./ sqrt_Ω
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
