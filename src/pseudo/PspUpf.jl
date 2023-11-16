using LinearAlgebra
using Interpolations: linear_interpolation
using PseudoPotentialIO: load_upf

struct PspUpf{T,I} <: NormConservingPsp
    ## From file
    Zion::Int          # Pseudo-atomic (valence) charge. UPF: `z_valence`
    lmax::Int          # Maximal angular momentum in the non-local part. UPF: `l_max`
    rgrid::Vector{T}   # Radial grid, can be linear or logarithmic. UPF: `PP_MESH/PP_R`
    drgrid::Vector{T}  # Radial grid derivative / integration factors. UPF: `PP_MESH/PP_RAB`
    vloc::Vector{T}    # Local part of the potential on the radial grid. UPF: `PP_LOCAL`
    # r^2 * β where β are Kleinman-Bylander non-local projectors on the radial grid.
    # UPF: `PP_NONLOCAL/PP_BETA.i`
    r2_projs::Vector{Vector{Vector{T}}}
    # Kleinman-Bylander energies. Stored per AM channel `h[l+1][i,j]`.
    # UPF: `PP_DIJ`
    h::Vector{Matrix{T}}
    # (UNUSED) Pseudo-wavefunctions on the radial grid. Can be used for wavefunction
    # initialization and as projectors for PDOS and DFT+U(+V).
    # r^2 * χ where χ are pseudo-atomic wavefunctions on the radial grid.
    # UPF: `PP_PSWFC/PP_CHI.i`
    r2_pswfcs::Vector{Vector{Vector{T}}}
    # (UNUSED) Occupations of the pseudo-atomic wavefunctions.
    # UPF: `PP_PSWFC/PP_CHI.i['occupation']`
    pswfc_occs::Vector{Vector{T}}
    # 4πr^2 ρion where ρion is the pseudo-atomic (valence) charge density on the
    # radial grid. Can be used for charge density initialization.
    # UPF: `PP_RHOATOM`
    r2_ρion::Vector{T}
    # r^2 ρcore where ρcore is the atomic core charge density on the radial grid,
    # used for non-linear core correction.
    # UPF: `PP_NLCC`
    r2_ρcore::Vector{T}

    ## Precomputed for performance
    # (USED IN TESTS) Local potential interpolator, stored for performance.
    vloc_interp::I
    # (USED IN TESTS) Projector interpolators, stored for performance.
    r2_projs_interp::Vector{Vector{I}}
    # (USED IN TESTS) Valence charge density interpolator, stored for performance.
    r2_ρion_interp::I
    # (USED IN TESTS) Core charge density interpolator, stored for performance.
    r2_ρcore_interp::I

    ## Extras
    rcut::T              # Radial cutoff for all quantities except pswfc.
                         # Used to avoid some numerical issues encountered when
                         # integrating over the full radial mesh.
    ircut::Int           # Index of the radial cutoff.
    identifier::String   # String identifying the pseudopotential.
    description::String  # Descriptive string. UPF: `comment`
end

"""
    PspUpf(path[, identifier])

Construct a Unified Pseudopotential Format pseudopotential from file.

Does not support:
- Non-linear core correction
- Fully-realtivistic / spin-orbit pseudos
- Bare Coulomb / all-electron potentials
- Semilocal potentials
- Ultrasoft potentials
- Projector-augmented wave potentials
- GIPAW reconstruction data
"""
function PspUpf(path; identifier=path, rcut=nothing)
    pseudo = load_upf(path)

    unsupported = []
    pseudo["header"]["has_so"] && push!(unsupported, "spin-orbit coupling")
    pseudo["header"]["pseudo_type"] == "SL" && push!(unsupported, "semilocal potential")
    pseudo["header"]["pseudo_type"] == "US" && push!(unsupported, "ultrasoft")
    pseudo["header"]["pseudo_type"] == "PAW" && push!(unsupported, "projector-augmented wave")
    pseudo["header"]["has_gipaw"] && push!(unsupported, "gipaw data")
    pseudo["header"]["pseudo_type"] == "1/r" && push!(unsupported, "Coulomb")
    length(unsupported) > 0 && error("Pseudopotential contains the following unsupported" *
                                     " features/quantities: $(join(unsupported, ","))")

    Zion = Int(pseudo["header"]["z_valence"])
    rgrid = pseudo["radial_grid"]
    drgrid = pseudo["radial_grid_derivative"]
    lmax = pseudo["header"]["l_max"]
    vloc = pseudo["local_potential"] ./ 2  # (Ry -> Ha)
    description = get(pseudo["header"], "comment", "")

    rcut = isnothing(rcut) ? last(rgrid) : rcut
    ircut = findfirst(>=(rcut), rgrid)

    # There are two possible units schemes for the projectors and coupling coefficients:
    # β [Ry Bohr^{-1/2}]  h [Ry^{-1}]
    # β [Bohr^{-1/2}]     h [Ry]
    # The quantity that's used in calculations is β h β, so the units don't practically
    # matter. However, HGH pseudos in UPF format use the first units, so we assume them
    # to facilitate comparison of the intermediate quantities with analytical HGH.

    r2_projs = map(0:lmax) do l
        betas_l = filter(beta -> beta["angular_momentum"] == l, pseudo["beta_projectors"])
        map(betas_l) do beta_li
            r_beta_ha = beta_li["radial_function"] ./ 2  # Ry -> Ha
            rgrid[1:length(r_beta_ha)] .* r_beta_ha  # rβ -> r²β
        end
    end

    h = Matrix[]
    count = 1
    for l = 0:lmax
        nproj_l = length(r2_projs[l+1])
        Dij_l = pseudo["D_ion"][count:count+nproj_l-1, count:count+nproj_l-1] .* 2  # 1/Ry -> 1/Ha
        push!(h, Dij_l)
        count += nproj_l
    end

    r2_pswfcs = map(0:lmax-1) do l
        pswfcs_l = filter(pseudo["atomic_wave_functions"]) do pswfc
            pswfc["angular_momentum"] == l
        end
        map(pswfcs_l) do pswfc_li
            r_pswfc_ha = pswfc_li["radial_function"] ./ 2  # Ry -> Ha
            rgrid[1:length(r_pswfc_ha)] .* r_pswfc_ha  # rχ -> r²χ
        end
    end

    pswfc_occs = map(0:lmax-1) do l
        pswfcs_l = filter(pseudo["atomic_wave_functions"]) do pswfc
            pswfc["angular_momentum"] == l
        end
        map(pswfc -> pswfc["occupation"], pswfcs_l)
    end

    r2_ρion = pseudo["total_charge_density"] ./ (4π)

    if pseudo["header"]["core_correction"]
        r2_ρcore = rgrid .^ 2 .* pseudo["core_charge_density"]
    else
        r2_ρcore = zeros(Float64, length(rgrid))
    end

    vloc_interp = linear_interpolation((rgrid,), vloc)
    r2_projs_interp = map(r2_projs) do r2_projs_l
        map(r2_projs_l) do r2_proj
            ir_cut = lastindex(r2_proj)
            linear_interpolation((rgrid[1:ir_cut],), r2_proj)
        end
    end
    r2_ρion_interp = linear_interpolation((rgrid,), r2_ρion)
    r2_ρcore_interp = linear_interpolation((rgrid,), r2_ρcore)

    PspUpf{eltype(rgrid),typeof(vloc_interp)}(Zion, lmax, rgrid, drgrid, vloc,
        r2_projs, h, r2_pswfcs, pswfc_occs, r2_ρion, r2_ρcore, vloc_interp,
        r2_projs_interp, r2_ρion_interp, r2_ρcore_interp, rcut, ircut,
        identifier, description
    )
end

charge_ionic(psp::PspUpf) = psp.Zion
has_valence_density(psp::PspUpf) = !all(iszero, psp.r2_ρion)
has_core_density(psp::PspUpf) = !all(iszero, psp.r2_ρcore)

function eval_psp_projector_real(psp::PspUpf, i, l, r::T)::T where {T<:Real}
    psp.r2_projs_interp[l+1][i](r) / r^2
end

function eval_psp_projector_fourier(psp::PspUpf, i, l, q::T)::T where {T<:Real}
    # The projectors may have been cut off before the end of the radial mesh
    # by PseudoPotentialIO because UPFs list a radial cutoff index for these
    # functions after which they are strictly zero in the file.
    ircut = min(psp.ircut, length(psp.r2_projs[l+1][i]))
    rgrid = @view psp.rgrid[1:ircut]
    r2_proj = psp.r2_projs[l+1][i][1:ircut]
    return hankel(rgrid, r2_proj, l, q)
end

function eval_psp_pswfc_real(psp::PspUpf, i, l, r::T)::T where {T<:Real}
    psp.r2_pswfcs_interp[l+1][i](r) / r^2
end

function eval_psp_pswfc_fourier(psp::PspUpf, i, l, q::T)::T where {T<:Real}
    # Pseudo-atomic wavefunctions are _not_ currently cut off like the other
    # quantities. They are the reason that PseudoDojo UPF files have a much
    # larger radial grid than their psp8 counterparts.
    # If issues arise, try cutting them off too.
    ircut = length(psp.r2_pswfcs[l+1][i])
    rgrid = @view psp.rgrid[1:ircut]
    return hankel(rgrid, psp.r2_pswfcs[l+1][i], l, q)
end

eval_psp_local_real(psp::PspUpf, r::T) where {T<:Real} = psp.vloc_interp(r)

function eval_psp_local_fourier(psp::PspUpf, q::T)::T where {T<:Real}
    # QE style C(r) = -Zerf(r)/r Coulomb tail correction used to ensure
    # exponential decay of `f` so that the Hankel transform is accurate.
    # H[Vloc(r)] = H[Vloc(r) - C(r)] + H[C(r)],
    # where H[-Zerf(r)/r] = -Z/q^2 exp(-q^2 /4)
    # ABINIT uses a more 'pure' Coulomb term with the same asymptotic behavior
    # C(r) = -Z/r; H[-Z/r] = -Z/q^2
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc = @view psp.vloc[1:psp.ircut]
    f = (
        rgrid .* (rgrid .* vloc .- -psp.Zion * erf.(rgrid))
        .* sphericalbesselj_fast.(0, q .* rgrid)
    )
    I = trapezoidal(rgrid, f)
    4T(π) * (I + -psp.Zion / q^2 * exp(-q^2 / T(4)))
end

function eval_psp_density_valence_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_ρion_interp(r) / r^2
end

function eval_psp_density_valence_fourier(psp::PspUpf, q::T) where {T<:Real}
    rgrid = @view psp.rgrid[1:psp.ircut]
    r2_ρion = @view psp.r2_ρion[1:psp.ircut]
    return hankel(rgrid, r2_ρion, 0, q)
end

function eval_psp_density_core_real(psp::PspUpf, r::T) where {T<:Real}
    psp.r2_ρcore_interp(r) / r^2
end

function eval_psp_density_core_fourier(psp::PspUpf, q::T) where {T<:Real}
    rgrid = @view psp.rgrid[1:psp.ircut]
    r2_ρcore = @view psp.r2_ρcore[1:psp.ircut]
    return hankel(rgrid, r2_ρcore, 0, q)
end

function eval_psp_energy_correction(T, psp::PspUpf, n_electrons)
    rgrid = @view psp.rgrid[1:psp.ircut]
    vloc = @view psp.vloc[1:psp.ircut]
    f = rgrid .* (rgrid .* vloc .- -psp.Zion)
    4T(π) * n_electrons * trapezoidal(rgrid, f)
end
