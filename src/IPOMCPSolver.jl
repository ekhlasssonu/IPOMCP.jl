abstract type AbstractIPOMCPSolver <: AbstractIPOMDPSolver end
type IPOMCPSolver <: AbstractIPOMCPSolver
    solvers::Vector{Tuple{POMCPSolver, Int64, Float64}}  #Solver, n_particles, qr constant
end

function getsolver(ipomcp_solver::IPOMCPSolver, level::Int64)
    return ipomcp_solver.solvers[level+1][1]
end

function n_particles(ipomcp_solver::IPOMCPSolver, level::Int64)
    return ipomcp_solver.solvers[level+1][2]
end

function qr_constant(ipomcp_solver::IPOMCPSolver, level::Int64)
    return ipomcp_solver.solvers[level+1][3]
end
