module IPOMCP

# package code goes here
import Base: ==, +, *, -, <, >, copy, Random, hash, length, rand, convert, print, println,haskey
importall POMDPs, ParticleFilters, BasicPOMCP
using POMDPToolbox,POMDPModels,QMDP
using DataFrames
using PmapProgressMeter
using ProgressMeter
using MCTS
using Parameters # for @with_kw
using AutoHashEquals
using StaticArrays
using JLD
using CPUTime
using Colors


export tester, test_intersection_problem, test_pedestrian_problem

include("AbstractTypeDeclarations.jl")
include("Auxiliary.jl")
include("SubintentionalModels.jl")
include("IntentionalModels.jl")
include("I-POMDP.jl")
include("MultiagentTiger.jl")
include("Intersection.jl")
include("PedestrianCrossing.jl")
include("InteractiveStates.jl")
include("IPOMCPSolver.jl")
include("IPOMCPPlanner.jl")
include("NestedParticleFilter.jl")
include("TreeFunctions.jl")
include("SimulationTools.jl")
include("I-POMDPTrial.jl")
end # module
