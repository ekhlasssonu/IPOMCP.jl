abstract type AbstractIPOMCPSolver <: AbstractPOMCPSolver end

type IPOMCPSolver <: AbstractIPOMCPSolver
    solvers::Vector{Tuple{POMCPSolver, Int64, Float64}} #Solver, n_particles, qr constant
end
