module IPOMCP

# package code goes here
import Base: ==, +, *, -, <, >, copy, Random, hash, length, rand, convert, print, println
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


export tester

include("Auxiliary.jl")
include("I-POMDPTrial.jl")
include("SubintentionalModels.jl")
include("IntentionalModels.jl")
include("I-POMDP.jl")
include("MultiagentTiger.jl")
include("InteractiveStates.jl")
include("IPOMCPSolver.jl")
include("IPOMCPPlanner.jl")
include("NestedParticleFilter.jl")
include("TreeFunctions.jl")
end # module
