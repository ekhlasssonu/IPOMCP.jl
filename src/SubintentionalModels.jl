abstract type Subintentional_Frame <: Frame end
abstract type Subintentional_Model <: Model end


abstract type Subintentional_Solver <: Solver end
abstract type Subintentional_Planner <: Policy end

type Static_Distribution_Frame{A} <: Subintentional_Frame
    agentID::Int64
    actProb::Dict{A, Float64}
end
==(a::Static_Distribution_Frame, b::Static_Distribution_Frame) = a.agentID == b.agentID && a.actProb == b.actProb
Base.hash(a::Static_Distribution_Frame, h::UInt64=zero(UInt64)) = hash((a.agentID,a.actProb),h)

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
Base.hash(a::Static_Distribution_Model, h::UInt64=zero(UInt64)) = hash((a.frame,a.history),h)

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

function update_model(i_frame::Frame, sm::Static_Distribution_Model, s::Any,
                        a::Any, oa::Any, sp::Any, rng::AbstractRNG, x...)
    return Dict{Model,Float64}(sm=>1.0)   #with prob
end


# FSM as sub intentional model
type FSMNode{A,O}
    actProb::Dict{A,Float64}
    tranProb::Dict{Tuple{A,O},Vector{Tuple{Int64,Float64}}}
end
==(a::FSMNode, b::FSMNode) = a.actProb == b.actProb && a.tranProb == b.tranProb
Base.hash(a::FSMNode, h::UInt64=zero(UInt64)) = hash((a.actProb,a.tranProb),h)

type FSM_Frame{A,O} <: Subintentional_Frame
    thisPOMDP::POMDP
    node_vec::Vector{FSMNode{A,O}}
end
==(a::FSM_Frame, b::FSM_Frame) = a.thisPOMDP == b.thisPOMDP && a.node_vec == b.node_vec
Base.hash(a::FSM_Frame, h::UInt64=zero(UInt64)) = hash((a.thisPOMDP,a.node_vec),h)
function FSM_Frame{A,O}(thisPOMDP::POMDP, size::Int64) where A where O
    return FSM_Frame{A,O}(thisPOMDP, Vector{FSMNode{A,O}}(size))
end

agentID(sf::FSM_Frame) = sf.thisPOMDP.agID

type FSM_Model{A,O} <: Subintentional_Model
    frame::FSM_Frame{A,O}
    history::Int    #CurrNode Idx

    FSM_Model{A,O}(frame::FSM_Frame{A,O}, h::Int) where A where O = new(frame, h)
end
==(a::FSM_Model, b::FSM_Model) = a.frame == b.frame && a.history == b.history
Base.hash(a::FSM_Model, h::UInt64=zero(UInt64)) = hash((a.frame,a.history),h)

sample_model{A,O}(frame::FSM_Frame{A,O}, rng::AbstractRNG=Base.GLOBAL_RNG) = FSM_Model{A,O}(frame, rand(rng, 1:length(frame.node_vec)))

function update_history{A,O}(sm::FSM_Model{A,O}, a::A, o::O, rng::AbstractRNG=Base.GLOBAL_RNG)
    currNode = sm.frame.node_vec[sm.history]
    nextnodeprob = get(currNode.tranProb,(A,O), 0)
    if nextnodeprob != 0
        sm.history = rand(rng,nextnodeprob)
    end
end

#TODO: Need the observation function, hack for now. Ideally add observation function to frame
function update_model{A,O}(i_frame::Frame, sm::FSM_Model{A,O}, s::Any,
                            a::A, oa::Any, sp::Any, rng::AbstractRNG, x...)
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



# FSM as sub intentional model
type FSMNode_Simple{A}
    actProb::Dict{A,Float64}
    tranProb::Dict{A,Vector{Tuple{Int64,Float64}}}
end
==(a::FSMNode_Simple, b::FSMNode_Simple) = a.actProb == b.actProb && a.tranProb == b.tranProb
Base.hash(a::FSMNode_Simple, h::UInt64=zero(UInt64)) = hash((a.actProb,a.tranProb),h)

type FSM_Simple_Frame{A} <: Subintentional_Frame
    agentID::Int64
    node_vec::Vector{FSMNode_Simple{A}}
end
==(a::FSM_Simple_Frame, b::FSM_Simple_Frame) = a.agentID == b.agentID && a.node_vec == b.node_vec
Base.hash(a::FSM_Simple_Frame, h::UInt64=zero(UInt64)) = hash((a.agentID,a.node_vec),h)
function FSM_Simple_Frame{A}(agentID::Int64, size::Int64) where A
    return FSM_Simple_Frame{A}(agentID, Vector{FSMNode_Simple{A}}(size))
end

agentID(sf::FSM_Simple_Frame) = sf.agID

type FSM_Simple_Model{A} <: Subintentional_Model
    frame::FSM_Simple_Frame{A}
    history::Int    #CurrNode Idx

    FSM_Simple_Model{A}(frame::FSM_Simple_Frame{A}, h::Int) where A = new(frame, h)
end
==(a::FSM_Simple_Model, b::FSM_Simple_Model) = a.frame == b.frame && a.history == b.history
Base.hash(a::FSM_Simple_Model, h::UInt64=zero(UInt64)) = hash((a.frame,a.history),h)

sample_model{A}(frame::FSM_Simple_Frame{A}, rng::AbstractRNG=Base.GLOBAL_RNG) = FSM_Simple_Model{A}(frame, rand(rng, 1:length(frame.node_vec)))

function update_history{A}(sm::FSM_Simple_Model{A}, a::A, rng::AbstractRNG=Base.GLOBAL_RNG)
    currNode = sm.frame.node_vec[sm.history]
    nextnodeprob = get(currNode.tranProb,A, 0)
    if nextnodeprob != 0
        sm.history = rand(rng,nextnodeprob)
    end
end

function update_model{A}(i_frame::Frame, sm::FSM_Simple_Model{A}, s::Any,
                            a::A, oa::Any, sp::Any, rng::AbstractRNG, x...)
    update_history(sm, a, rng)   #with prob
    return Dict{Model,Float64}(sm=>1.0)
end

type FSM_Simple_Solver <: Subintentional_Solver
    rng::AbstractRNG
end
solver(frame::FSM_Simple_Frame; rng::AbstractRNG = Base.GLOBAL_RNG) = FSM_Simple_Solver(rng)

type FSM_Simple_Planner <: Subintentional_Planner
    solver::FSM_Simple_Solver
    rng::AbstractRNG
    frame::FSM_Simple_Frame
end
solve(solver::FSM_Simple_Solver, frame::FSM_Simple_Frame) = FSM_Simple_Planner(solver, solver.rng, frame)   #Is there a better way?

function actionProb(planner::FSM_Simple_Planner, hist::Int64)
    return planner.frame.node_vec[hist].actProb
end

function action(planner::FSM_Simple_Planner, hist::Int64)
    rng = planner.rng
    actProb = actionProb(planner,hist)
    aj = rand(actProb, rng)
    return aj
end
