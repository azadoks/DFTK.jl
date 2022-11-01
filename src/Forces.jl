using OrderedCollections
"""
A simple struct to contain a vector of force vectors, and utilities to print them in a nice format.
"""
struct Forces{T <: Union{Nothing, AbstractVector}}
    # energies["TermName"]
    # parametrization on T acts as a nice check that all terms return correct type
    forces::OrderedDict{String, T}
end

function Base.show(io::IO, forces::Forces)
    print(io, "Forces(total = $(forces.total))")
end
function Base.show(io::IO, ::MIME"text/plain", forces::Forces)
    println(io, "Force breakdown (in Ha/Bohr):")
    for (name, value) in forces.forces
        if !isnothing(value)
            @printf io "    %-20s\n" string(name)
            for force_vector in value
                @printf io "        %10.7f %10.7f %10.7f\n" force_vector...
            end
        end
    end
    @printf io "    %-20s\n" "total"
    for force_vector in forces.total
        @printf io "        %10.7f %10.7f %10.7f\n" force_vector...
    end
    # @printf io "\n    %-20s%-15.12f" "total" forces.total
end
Base.getindex(forces::Forces, i) = forces.forces[i]
Base.values(forces::Forces)      = values(forces.forces)
Base.keys(forces::Forces)        = keys(forces.forces)
Base.pairs(forces::Forces)       = pairs(forces.forces)
Base.iterate(forces::Forces)     = iterate(forces.forces)
Base.iterate(forces::Forces, state) = iterate(forces.forces, state)
Base.haskey(forces::Forces, key) = haskey(forces.forces, key)

function Forces(term_types::Vector, forces::Vector{T}) where {T}
    # nameof is there to get rid of parametric types
    Forces{T}(OrderedDict([string(nameof(typeof(term))) => forces[i]
                             for (i, term) in enumerate(term_types)]...))
end

function Forces(scfres)
    term_forces = map(scfres.basis.terms) do term
        f = compute_forces(term, scfres.basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
        if isnothing(f)
            f
            # [zero(Vec3{eltype(scfres.basis)}) for _ in scfres.basis.model.positions]
        else
            covector_red_to_cart.(scfres.basis.model, f)
        end
    end
    Forces(scfres.basis.model.term_types, term_forces)
end

function Base.propertynames(forces::Forces, private::Bool=false)
    ret = keys(forces)
    append!(ret, "total")
    private && append!(ret, "forces")
end
function Base.getproperty(forces::Forces, x::Symbol)
    x == :total && return sum(f for f in values(forces) if !isnothing(f))
    x == :forces && return getfield(forces, x)
    forces.forces[string(x)]
end
