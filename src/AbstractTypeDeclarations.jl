abstract type Frame end
abstract type Model end

abstract type AbstractInteractiveState{S} end

abstract type AbstractIPOMDP <: Frame end
abstract type AbstractParticleInteractiveBelief{T} <: AbstractParticleBelief{T} end

abstract type AbstractIPOMDPSolver end
