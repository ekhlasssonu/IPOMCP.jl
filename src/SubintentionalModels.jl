abstract type Subintentional_Frame <: Frame end
abstract type Subintentional_Model <: Model end


abstract type Subintentional_Solver <: Solver end
abstract type Subintentional_Planner <: Policy end

type Static_Distribution_Frame{A} <: Subintentional_Frame
    agentID::Int64
    actProb::Dict{A, Float64}
end

agentID(sf::Subintentional_Frame) = sf.agentID

type Static_Distribution_Model{A} <: Subintentional_Model
    frame::Static_Distribution_Frame{A}
    history::Int

    Static_Distribution_Model{A}(frame::Static_Distribution_Frame{A}) where{A} = new(frame, 0)
end

function Static_Distribution_Model{A}(agentID::Int64, actProb::Dict{A, Float64})
    sum = 0.0
    for (act,prob) in actProb
        sum = sum + prob
    end
    for (act,prob) in actProb
        prob = prob/sum
    end
    return Static_Distribution_Model(Static_Distribution_Frame(agentID, actProb),0)
end
sample_model{A}(frame::Static_Distribution_Frame{A}, rng::AbstractRNG=Base.GLOBAL_RNG) = Static_Distribution_Model{A}(frame)

type Static_Distribution_Solver <: Subintentional_Solver
    rng::AbstractRNG
end

solver(frame::Static_Distribution_Frame; rng::AbstractRNG = Base.GLOBAL_RNG) = Static_Distribution_Solver(rng)
type Static_Distribution_Planner <: Subintentional_Planner
    solver::Static_Distribution_Solver
    rng::AbstractRNG
    frame::Static_Distribution_Frame
end

solve(solver::Static_Distribution_Solver, frame::Static_Distribution_Frame) = Static_Distribution_Planner(solver, solver.rng, frame)   #Is there a better way?

function action(planner::Static_Distribution_Planner, hist::Int64)
    rng = planner.rng
    actProb = planner.frame.actProb
    aj = rand(actProb, rng)
    return aj
end

update_history(sm::Static_Distribution_Model; a::Any=1, o::Any = 1) = sm

function update_model(sm::Static_Distribution_Model, s::Any, a::Any, oa::Any, sp::Any, rng::AbstractRNG, x...)
    return Dict{Model,Float64}(sm=>1.0)  #with prob
end
