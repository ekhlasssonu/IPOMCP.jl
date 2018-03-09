abstract type Subintentional_Frame <: Frame end
abstract type Subintentional_Model <: Model end


abstract type Subintentional_Solver <: Solver end
abstract type Subintentional_Planner <: Policy end

type Static_Distribution_Frame{A} <: Subintentional_Frame
    agentID::Int64
    actProb::Dict{A, Float64}
end
==(a::Static_Distribution_Frame, b::Static_Distribution_Frame) = a.agentID == b.agentID && a.actProb == b.actProb
Base.hash(a::Static_Distribution_Frame) = hash((a.agentID,a.actProb))

agentID(sf::Subintentional_Frame) = sf.agentID

type Static_Distribution_Model{A} <: Subintentional_Model
    frame::Static_Distribution_Frame{A}
    history::Int

    Static_Distribution_Model{A}(frame::Static_Distribution_Frame{A}) where{A} = new(frame, 0)
end
function print(model::Static_Distribution_Model, numTabs::Int64=0)
    for i in 1:numTabs
        print("\t")
    end
    println("[",model.frame.actProb,", ",model.history,"]")
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
==(a::Static_Distribution_Model, b::Static_Distribution_Model) = a.frame == b.frame && a.history == b.history
Base.hash(a::Static_Distribution_Model) = hash((a.frame,a.history))

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

function actionProb(planner::Static_Distribution_Planner, hist::Int64)
    return planner.frame.actProb
end

function action(planner::Static_Distribution_Planner, hist::Int64)
    rng = planner.rng
    actProb = planner.frame.actProb
    aj = rand(actProb, rng)
    return aj
end

update_history(sm::Static_Distribution_Model; a::Any=1, o::Any = 1) = sm

function update_model(sm::Static_Distribution_Model, s::Any, a::Any, oa::Any, sp::Any, rng::AbstractRNG, x...)
    return Dict{Model,Float64}(sm=>1.0)   #with prob
end


# FSM as sub intentional model
type FSMNode{A,O}
    actProb::Dict{A,Float64}
    tranProb::Dict{Tuple{A,O},Vector{Tuple{Int64,Float64}}}
end
==(a::FSMNode, b::FSMNode) = a.actProb == b.actProb && a.tranProb == b.tranProb
Base.hash(a::FSMNode) = hash((actProb,a.tranProb))

type FSM_Frame{A,O} <: Subintentional_Frame
    agentID::Int64
    node_vec::Vector{FSMNode{A,O}}
end
==(a::FSM_Frame, b::FSM_Frame) = a.agentID == b.agentID && a.node_vec == b.node_vec
Base.hash(a::FSM_Frame) = hash((a.agentID,a.node_vec))
function FSM_Frame{A,O}(agentID::Int64, size::Int64) where A where O
    return FSM_Frame{A,O}(agentID, Vector{FSMNode{A,O}}(size))
end

agentID(sf::FSM_Frame) = sf.agentID

type FSM_Model{A,O} <: Subintentional_Model
    frame::FSM_Frame{A,O}
    history::Int    #CurrNode Idx

    FSM_Model{A,O}(frame::FSM_Frame{A,O}) where A where O = new(frame, 1)
end
==(a::FSM_Model, b::FSM_Model) = a.frame == b.frame && a.history == b.history
Base.hash(a::FSM_Model) = hash((a.frame,a.history))

sample_model{A}(frame::FSM_Frame{A}, rng::AbstractRNG=Base.GLOBAL_RNG) = FSM_Model{A}(frame, rand(rng, 1:length(frame.node_vec)))

function update_history{A,O}(sm::FSM_Model{A,O}; a::A, o::A, rng::AbstractRNG=Base.GLOBAL_RNG)
    currNode = sm.frame.node_vec[sm.history]
    hist = rand(rng,get(currNode.tranProb,(A,O), 0))
    hist != 0 ? sm.history = hist : nothing
end

#TODO: Need the observation function, hack for now. Ideally add observation function to frame
function update_model(sm::Static_Distribution_Model, s::Any, a::Any, oa::Any, sp::Any, rng::AbstractRNG, x...)
    o = x[1]
    return update_history{A,O}(sm, a, o, rng)   #with prob
end

type FSM_Solver <: Subintentional_Solver
    rng::AbstractRNG
end
solver(frame::FSM_Frame; rng::AbstractRNG = Base.GLOBAL_RNG) = FSM_Solver(rng)

type FSM_Planner <: Subintentional_Planner
    solver::FSM_Solver
    rng::AbstractRNG
    frame::FSM_Frame
end
solve(solver::FSM_Solver, frame::FSM_Frame) = FSM_Planner(solver, solver.rng, frame)   #Is there a better way?

function actionProb(planner::FSM_Planner, hist::Int64)
    return planner.frame.node_vec[hist].actProb
end

function action(planner::FSM_Planner, hist::Int64)
    rng = planner.rng
    actProb = actionProb(planner,hist)
    aj = rand(actProb, rng)
    return aj
end
