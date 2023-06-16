function compute_form_factor_radials!(radial_work::AbstractArray{Complex{T},3},
                                      basis::PlaneWaveBasis{T}, kpt::Kpoint, el::Element,
                                      quantity_flag::PseudoPotentialIOExperimental.ProjectorFlag) where {T}
    qs_cart = norm.(Gplusk_vectors_cart(basis, kpt))
    for l in angular_momenta(el), n in 1:n_radials(el, quantity_flag, l)
        quantity = get_quantity(el, quantity_flag, l, n)
        radial_work[:,n,l+1] .= quantity.itp.(qs_cart)
    end
    radial_work
end

function build_projection_vectors!(P::AbstractArray{Complex{T}},
                                   structure_factor_work::AbstractVector{Complex{T}},
                                   radial_work::AbstractArray{Complex{T}},
                                   angular_work::AbstractVector{Complex{T}},
                                   basis::PlaneWaveBasis{T},
                                   kpt::Kpoint, el::Element, position::Vec3{T},
                                   quantity_flag::PseudoPotentialIOExperimental.ProjectorFlag) where {T}
    qs_cart = Gplusk_vectors_cart(basis, kpt)
    structure_factor_work .= cis2pi.(-dot.(Gplusk_vectors(basis, kpt), Ref(position)))

    i_proj = 1
    for l in angular_momenta(el), m in (-l):(+l)
        angular_work .= (-im)^l .* ylm_real.(l, m, qs_cart)
        for n in 1:n_radials(el, quantity_flag, l)
            P[:,i_proj] .= structure_factor_work                # Copy over the structure factors
            P[:,i_proj] .*= angular_work                        # Angular part of the form factors
            P[:,i_proj] .*= radial_work[:, n, l+1]              # Radial part of the form factors
            P[:,i_proj] ./= sqrt(basis.model.unit_cell_volume)  # Normalization
            i_proj += 1
        end
    end
    P
end

function build_projection_vectors!(P::AbstractArray{Complex{T}}, basis::PlaneWaveBasis{T},
                                   kpt::Kpoint, psps::AbstractVector{Element},
                                   psp_positions::AbstractVector{Vec3{T}},
                                   quantity_flag::PseudoPotentialIOExperimental.ProjectorFlag) where {T}
    # Allocate working arrays for the angular and radial parts of the form factors
    max_l = maximum(max_angular_momentum, psps)
    max_n_radials = maximum(psps) do specie
        maximum(angular_momenta(specie)) do l
            n_radials(specie, quantity_flag, l)
        end
    end
    radial_work = zeros_like(P, size(P, 1), max_n_radials, max_l + 1)
    angular_work = zeros_like(P, size(P, 1))
    structure_factor_work = zeros_like(P, size(P, 1))

    i_proj = 1
    for (specie, positions) in zip(psps, psp_positions)
        n_specie_projs = n_angulars(specie, quantity_flag)
        compute_form_factor_radials!(radial_work, basis, kpt, specie, quantity_flag)
        for position in positions
            build_projection_vectors!(@view(P[:,i_proj:(i_proj + n_specie_projs - 1)]),
                                      structure_factor_work, radial_work, angular_work,
                                      basis, kpt, specie, position,
                                      quantity_flag)
            i_proj += n_specie_projs
        end
    end
    return P
end

function build_projection_vectors(basis::PlaneWaveBasis{T}, kpt::Kpoint, psps::AbstractVector{Element},
                                  psp_positions::AbstractVector{AbstractVector{Vec3{T}}},
                                  quantity_flag::PseudoPotentialIOExperimental.ProjectorFlag) where {T}
    n_q = length(Gplusk_vectors(basis, kpt))
    n_proj = sum(zip(psps, psp_positions)) do (psp, positions)
        n_angulars(psp, quantity_flag) * length(positions)
    end
    P = zeros_like(Gplusk_vectors(basis, kpt), Complex{T}, n_q, n_proj)
    return build_projection_vectors!(P, basis, kpt, psps, species_positions, quantity_flag)
end
