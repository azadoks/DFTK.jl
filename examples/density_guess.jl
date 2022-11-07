# # Density Guesses
#
# We compare four different approaches to charge density initialization for starting a
# density-based SCF.

# First we set up our problem
using DFTK
using LinearAlgebra

# We use a numeric norm-conserving PSP in UPF format from the
# [PseudoDojo](http://www.pseudo-dojo.org/) v0.4 scalar-relativistic LDA standard stringency
# family because it containts valence charge densities which can be used for a more
# tailored density guess.
PSEUDOLIB = "https://raw.githubusercontent.com/JuliaMolSim/PseudoLibrary"
URL_UPF = PSEUDOLIB * "/main/pseudos/pd_nc_sr_lda_standard_04_upf/Si.upf";

function silicon_scf(method)
    a = 10.26  # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                    [1 0 1.];
                    [1 1 0.]]
    Si = ElementPsp(:Si; psp=load_psp(URL_UPF))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    model = model_LDA(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut=12, kgrid=[4, 4, 4])

    ρguess = guess_density(basis, method)

    is_converged = DFTK.ScfConvergenceEnergy(1e-10)
    self_consistent_field(basis; is_converged, ρ=ρguess)
end;

# ## Random guess
# The random density is normalized to the number of electrons provided.
scfres_random = silicon_scf(RandomGuessDensity(8.0));

# ## Superposition of Gaussian densities
# The Gaussians are defined by a tabulated atom decay length.
scfres_gaussian = silicon_scf(GaussianGuessDensity());

# ## Superposition of pseudopotential valence charge densities
# This method only works when _all_ atoms are `ElementPsp`s and _all_ of the
# pseudopotentials contain valence charge densities. If only some of the
# pseudopotentials have valence charge densities, use the `AutoGuessDensity()` method
# which uses Gaussian densities for atoms with pseudopotentials that don't have
# valence charge densities.
scfres_psp = silicon_scf(PspGuessDensity());
