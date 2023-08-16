using Random  # Used to have a generic API for CPU and GPU computations alike: see random_orbitals

# Returns the occupied orbitals, the occupation array and optionally the eigenvalues without
# virtual states (or states with small occupation level for metals).
# threshold is a parameter to distinguish between states we want to keep and the
# others when using temperature. It is set to 0.0 by default, to treat with insulators.
function select_occupied_orbitals(basis, ψ, occupation; threshold=0.0)
    N = [something(findlast(x -> x > threshold, occk), 0) for occk in occupation]
    selected_ψ   = [@view ψk[:, 1:N[ik]] for (ik, ψk)   in enumerate(ψ)]
    selected_occ = [      occk[1:N[ik]]  for (ik, occk) in enumerate(occupation)]

    # if we have an insulator, sanity check that the orbitals we kept are the
    # occupied ones
    if iszero(threshold)
        model   = basis.model
        n_spin  = model.n_spin_components
        n_bands = div(model.n_electrons, n_spin * filled_occupation(model), RoundUp)
        @assert n_bands == size(selected_ψ[1], 2)
    end
    (; ψ=selected_ψ, occupation=selected_occ)
end

# Packing routines used in direct_minimization and newton algorithms.
# They pack / unpack sets of ψ's (or compatible arrays, such as hamiltonian
# applies and gradients) to make them compatible to be used in algorithms
# from IterativeSolvers.
# Some care is needed here : some operators (for instance K in newton.jl)
# are real-linear but not complex-linear. To overcome this difficulty, instead of
# seeing them as operators from C^N to C^N, we see them as
# operators from R^2N to R^2N. In practice, this is done with the
# reinterpret function from julia.
# /!\ pack_ψ does not share memory while unpack_ψ does

reinterpret_real(x) = reinterpret(real(eltype(x)), x)
reinterpret_complex(x) = reinterpret(Complex{eltype(x)}, x)

function pack_ψ(ψ)
    # TODO as an optimization, do that lazily? See LazyArrays
    vcat([vec(ψk) for ψk in ψ]...)
end

# Returns pointers into the unpacked ψ
# /!\ The resulting array is only valid as long as the original x is still in live memory.
function unsafe_unpack_ψ(x, sizes_ψ)
    lengths = prod.(sizes_ψ)
    ends = cumsum(lengths)
    # We unsafe_wrap the resulting array to avoid a complicated type for ψ.
    map(1:length(sizes_ψ)) do ik
        unsafe_wrap(Array{complex(eltype(x))},
                    pointer(@views x[ends[ik]-lengths[ik]+1:ends[ik]]),
                    sizes_ψ[ik])
    end
end
unpack_ψ(x, sizes_ψ) = deepcopy(unsafe_unpack_ψ(x, sizes_ψ))

function random_orbitals(basis::PlaneWaveBasis{T}, kpt::Kpoint, howmany) where {T}
    @static if VERSION < v"1.7"  # TaskLocalRNG not yet available.
        orbitals = randn(Complex{T}, length(G_vectors(basis, kpt)), howmany)
        orbitals = to_device(basis.architecture, orbitals)
    else
        orbitals = similar(basis.G_vectors, Complex{T}, length(G_vectors(basis, kpt)), howmany)
        randn!(TaskLocalRNG(), orbitals)  # use the RNG on the device if we're using a GPU
    end
    ortho_qr(orbitals)
end

function guess_orbitals(basis::PlaneWaveBasis{T}, kpt::Kpoint, howmany) where {T}
    model = basis.model

    atom_groups = [group for group in model.atom_groups
                   if has_orbitals(model.atoms[first(group)])]
    atom_positions = [model.positions[group] for group in atom_groups]
    atoms = [basis.fourier_atoms[first(group)] for group in atom_groups]
    atom_orbitals = [atom.potential.orbitals for atom in atoms]

    n_orbitals = sum(zip(atom_orbitals, atom_positions)) do (orbitals, positions)
        n_angulars = sum(orbital -> 2 * angular_momentum(orbital) + 1, orbitals)
        n_angulars * length(positions)
    end

    orbitals = similar(basis.G_vectors, Complex{T}, length(G_vectors(basis, kpt)), howmany)

    if howmany < n_orbitals
        # Sort the states by their pseudo energy (if available) then build
        error("Not yet supported")
    elseif howmany == n_orbitals
        # Just call build_projection_vectors
        build_projection_vectors!(orbitals, basis, kpt, atom_orbitals, atom_positions)
    else
        # Call build_projection vectors to make all the pseudo-atomic states, then
        # fill in the rest with random states
        build_projection_vectors!(
            @view(orbitals[:,1:n_orbitals]),
            basis,
            kpt,
            atom_orbitals,
            atom_positions
        )
        randn!(TaskLocalRng(), @view(orbitals[:,n_orbitals+1:end]))
    end
    ortho_qr(orbitals)
end
