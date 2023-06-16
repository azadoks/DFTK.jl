function atomic_density!(ρ::AbstractArray{Complex{T}}, basis::PlaneWaveBasis{T},
                         atom::Element,
                         position::AbstractVector{T},
                         quantity_flag::PseudoPotentialIOExperimental.DensityFlag;
                         coefficient::T=one(T)) where {T}
    quantity = PseudoPotentialIOExperimental.get_quantity(atom, quantity_flag)
    ρ .= cis2pi.(-dot.(G_vectors(basis), Ref(position)))  # Structure factors
    ρ .*= quantity.itp.(norm.(G_vectors_cart(basis)))     # Form factors
    ρ .*= coefficient                                     # Often magnetic moments
    ρ ./= sqrt(basis.model.unit_cell_volume)              # Normalization
end

function atomic_density(basis::PlaneWaveBasis{T}, atom::Element, position::AbstractVector{T},
                        quantity::PseudoPotentialIOExperimental.DensityFlag;
                        coefficient::T=one(T)) where {T}
    ρ = zeros_like(G_vectors(basis), Complex{T})
    atomic_density!(ρ, basis, atom, position, quantity; coefficient)
end

function atomic_density_superposition(basis::PlaneWaveBasis{T},
                                      quantity::PseudoPotentialIOExperimental.DensityFlag;
                                      coefficients::AbstractVector{T}=ones(T, length(atoms))) where {T}
    atoms = basis.fourier_atoms
    positions = model.positions
    ρ = zeros_like(G_vectors(basis), Complex{T})
    ρ_atom = zeros_like(G_vectors(basis), Complex{T})

    for (atom, position, coefficient) in zip(atoms, positions, coefficients)
        atomic_density!(ρ_atom, basis, atom, position, quantity; coefficient)
        ρ .+= ρ_atom
    end

    enforce_real!(basis, ρ)
    irfft(basis, ρ)
end
