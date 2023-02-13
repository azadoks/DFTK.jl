abstract type DensityMethod                  end
abstract type AtomicDensity <: DensityMethod end

struct RandomDensity           <: DensityMethod end
struct CoreDensity             <: AtomicDensity end
struct ValenceGaussianDensity  <: AtomicDensity end
struct ValenceNumericalDensity <: AtomicDensity end
struct ValenceAutoDensity      <: AtomicDensity end

# Random density method
function guess_density(basis::PlaneWaveBasis, ::RandomDensity;
                       n_electrons=basis.model.n_electrons)
    random_density(basis, n_electrons)
end

function random_density(basis::PlaneWaveBasis{T}, n_electrons) where {T}
    ρtot  = rand(T, basis.fft_size)
    ρtot  = ρtot .* n_electrons ./ (sum(ρtot) * basis.dvol)  # Integration to n_electrons
    ρspin = nothing
    if basis.model.n_spin_components > 1
        ρspin = rand((-1, 1), basis.fft_size) .* rand(T, basis.fft_size) .* ρtot
        @assert all(abs.(ρspin) .≤ ρtot)
    end
    ρ_from_total_and_spin(ρtot, ρspin)
end

# Atomic density methods
function guess_density(basis::PlaneWaveBasis, magnetic_moments=[],
                       n_electrons=basis.model.n_electrons)
    atomic_density(basis, ValenceAutoDensity(), magnetic_moments, n_electrons)
end

function guess_density(basis::PlaneWaveBasis, method::AtomicDensity, magnetic_moments=[];
                       n_electrons=basis.model.n_electrons)
    atomic_density(basis, method, magnetic_moments, n_electrons)
end

function atomic_density(basis::PlaneWaveBasis, method::AtomicDensity, magnetic_moments,
                        n_electrons)
    ρtot = atomic_total_density(basis, method)
    ρspin = atomic_spin_density(basis, method, magnetic_moments)
    ρ = ρ_from_total_and_spin(ρtot, ρspin)
    
    N = sum(ρ) * basis.model.unit_cell_volume / prod(basis.fft_size)

    if !isnothing(n_electrons) && (N > 0)
        ρ .*= n_electrons / N  # Renormalize to the correct number of electrons
    end
    ρ
end

function atomic_total_density(basis::PlaneWaveBasis{T}, method::AtomicDensity;
                              coefficients=ones(T, length(basis.model.atoms))) where {T}
    form_factors = atomic_density_form_factors(basis, method)
    atomic_density_superposition(basis, form_factors; coefficients)
end

function atomic_spin_density(basis::PlaneWaveBasis{T}, method::AtomicDensity,
                             magnetic_moments) where {T}
    model = basis.model
    if model.spin_polarization in (:none, :spinless)
        isempty(magnetic_moments) && return nothing
        error("Initial magnetic moments can only be used with collinear models.")
    end

    # If no magnetic moments start with a zero spin density
    magmoms = Vec3{T}[normalize_magnetic_moment(magmom) for magmom in magnetic_moments]
    if all(iszero, magmoms)
        @warn("Returning zero spin density guess, because no initial magnetization has " *
              "been specified in any of the given elements / atoms. Your SCF will likely " *
              "not converge to a spin-broken solution.")
        return zeros(T, basis.fft_size)
    end

    @assert length(magmoms) == length(basis.model.atoms)
    coefficients = map(zip(basis.model.atoms, magmoms)) do (atom, magmom)
        iszero(magmom[1:2]) || error("Non-collinear magnetization not yet implemented")
        magmom[3] ≤ n_elec_valence(atom) || error(
            "Magnetic moment $(magmom[3]) too large for element $(atomic_symbol(atom)) " *
            "with only $(n_elec_valence(atom)) valence electrons."
        )
        magmom[3] / n_elec_valence(atom)
    end

    form_factors = atomic_density_form_factors(basis, method)
    atomic_density_superposition(basis, form_factors; coefficients)    
end

function atomic_density_superposition(basis::PlaneWaveBasis{T},
                                      form_factors::IdDict{Tuple{Int,T},T};
                                      coefficients=ones(T, length(basis.model.atoms))
                                      )::Array{T,3} where {T}
    model = basis.model
    G_cart = G_vectors_cart(basis)

    ρ = map(enumerate(G_vectors(basis))) do (iG, G)
        Gnorm = norm(G_cart[iG])
        ρ_iG = sum(enumerate(model.atom_groups); init=zero(Complex{T})) do (igroup, group)
            sum(group) do iatom
                structure_factor::Complex{T} = cis2pi(-dot(G, model.positions[iatom]))
                coefficients[iatom]::T * form_factors[(igroup, Gnorm)]::T * structure_factor
            end
        end
        ρ_iG / sqrt(model.unit_cell_volume)
    end
    enforce_real!(basis, ρ)  # Symmetrize Fourier coeffs to have real iFFT
    irfft(basis, ρ)
end

function atomic_density_form_factors(basis::PlaneWaveBasis{T},
                                     method::AtomicDensity
                                     )::IdDict{Tuple{Int,T},T} where {T<:Real}
    model = basis.model
    form_factors = IdDict{Tuple{Int,T},T}()  # IdDict for Dual compatability
    for G in G_vectors_cart(basis)
        Gnorm = norm(G)
        for (igroup, group) in enumerate(model.atom_groups)
            if !haskey(form_factors, (igroup, Gnorm))
                element = model.atoms[first(group)]
                form_factor = atomic_density(element, Gnorm, method)
                form_factors[(igroup, Gnorm)] = form_factor
            end
        end
    end
    form_factors
end

function atomic_density(element::Element, Gnorm::T,
                        ::ValenceGaussianDensity)::T where {T <: Real}
    gaussian_valence_charge_density_fourier(element, Gnorm)
end

function atomic_density(element::Element, Gnorm::T,
                        ::ValenceNumericalDensity)::T where {T <: Real}
    eval_psp_density_valence_fourier(element.psp, Gnorm)
end

function atomic_density(element::Element, Gnorm::T,
                        ::ValenceAutoDensity)::T where {T <: Real}
    valence_charge_density_fourier(element, Gnorm)
end

function atomic_density(element::Element, Gnorm::T,
                        ::CoreDensity)::T where {T <: Real}
    has_density_core(element) ? core_charge_density_fourier(element, Gnorm) : zero(T)
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
        [2.0, 1.6, 1.25, 1.1, 1.0, 0.9, 0.8, 0.7 , 0.7, 0.7, 0.6]
    elseif n_elec_core < 12.5
        # Mg core (1s2 2s2 2p6 3s2): Adjusted on 19k, and on n_elec_core==10
        [1.9, 1.5, 1.15, 1.0, 0.9, 0.8, 0.7, 0.6 , 0.6, 0.6, 0.5]
    elseif n_elec_core < 18.5
        # Ar core (Ne + 3s2 3p6): Adjusted on 20ca, 25mn and 30zn
        [2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.65, 0.6]
    elseif n_elec_core < 28.5
        # Full 3rd shell core (Ar + 3d10): Adjusted on 31ga, 34se and 38sr
        [1.5, 1.25, 1.15, 1.05, 1.00, 0.95, 0.95, 0.9, 0.9, 0.85, 0.85, 0.80,
         0.8 , 0.75, 0.7]
    elseif n_elec_core < 36.5
        # Krypton core (Ar + 3d10 4s2 4p6): Adjusted on 39y, 42mo and 48cd
        [2.0, 2.00, 1.60, 1.40, 1.25, 1.10, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.7]
    else
        # For the remaining elements, consider a function of n_elec_valence only
        [2.0 , 2.00, 1.55, 1.25, 1.15, 1.10, 1.05, 1.0 , 0.95, 0.9, 0.85, 0.85, 0.8]
    end
    data[min(n_elec_valence, length(data))]
end
atom_decay_length(sp::Element) = atom_decay_length(n_elec_core(sp), n_elec_valence(sp))