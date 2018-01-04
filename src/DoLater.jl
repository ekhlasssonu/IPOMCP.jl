#TODO: Add observation function and obs weight functions
type FSMNode{A,O}
    actProb::Dict{A, Float64}
    transition::Dict{Tuple{A,O}, FSMNode}
end
function FSMNode{A,O}(aProb::Dict{A,Float64},trn::Dict{Tuple{A,O}, FSMNode})
    sum = 0.0
    for (act,prob) in aProb
        sum = sum + prob
    end
    for (act,prob) in aProb
        prob = prob/sum
    end
    return FSMNode{A,O}(aProb, trn)
end

function get_parameter_type{A,O}(fsmNode::FSMNode{A,O})
    return A,O
end

#Copy constructor
function FSMNode(anotherNode::FSMNode)
    A,O = get_parameter_type(anotherNode)
    actProb = Dict{A, Float64}()
    transition = Dict{Tuple{A,O}, FSMNode}()
    for (k,v) in anotherNode.actProb
        actProb[k] = v
    end
    for (k,v) in anotherNode.transition
        transition[k] = v
    end
    return FSMNode{A,O}(actProb, transition)
end

type FSM_Frame{A,O} <: Subintentional_Frame
    ag_ID::Int64
    nodeSet::Vector{FSMNode{A,O}}
end

type FSM_Model{A,O} <: Subintentional_Model
    frame::FSM_Frame{A,O}
    history::FSMNode{A,O} #Current node
end

FSM_Model{A,O}(frame::FSM_Frame{A,O}) = FSM_Model(frame, frame.nodeSet[1])
FSM_Model{A,O}(agentID::Int64, nodeSet::Vector{FSMNode{A,O}}) = FSM_Model(FSM_Frame{A,O}(agentID, nodeSet), nodeSet[1])

function rand(fsm_model::FSM_Model; rng::AbstractRNG=Base.GLOBAL_RNG)
    fsm_model.history = rand(rng, fsm_model.frame.nodeSet)
end
function solve(sm::FSM_Model, rng::AbstractRNG=Base.GLOBAL_RNG)
    rnd = rand(rng)
    currNode = sm.history
    return currNode.actProb
end
function update_history{A,O}(sm::FSM_Model{A,O}, a::A, o::O)
    currNode = sm.history
    nextNode = get(currNode.transition, (a,o), currNode)
    sm.history = nextNode
end
function generate_modelp{A,OA}(sm::FSM_Model, sp, a, oa, rng::AbstractRNG=Base.GLOBAL_RNG)
    # a is the action of the agent  whose IS the model sm is a part of
    # Hence the agent whose model is being updated performs oa
    #First generate observation
    oag_ID = sm.frame.ag_ID
    if oag_ID == 1
        a1 = oa
        a2 = a
    else
        a1 = a
        a2 = oa
    end

    o = generate_o(sm, sp, (a1,a2), rng) #The agent who is recieving obs performs oa.

    return update_history(sm, oa, o),1.0
end

# Goes in Tiger POMDP
function initialize_fsm(::Union{Level_0_tigerPOMDP, Level_l_tigerPOMDP})
    nodeSet = Vector{FSMNode}(5)
    for i in 1:length(nodeSet)
        nodeSet[i] = FSMNode{Int64, Tuple{Int64,Int64}}(Dict{Int64, Float64}(), Dict{Int64, Tuple{Int64,Int64}}())
    end
    nodeSet[1].actProb[OL] = 1.0
    nodeSet[2].actProb[L] = 1.0
    nodeSet[3].actProb[L] = 1.0
    nodeSet[4].actProb[L] = 1.0
    nodeSet[5].actProb[OR] = 1.0

    nodeSet[3].transition[(L,(GL,CL))] = nodeSet[4]
    nodeSet[3].transition[(L,(GL,SL))] = nodeSet[4]
    nodeSet[3].transition[(L,(GL,CR))] = nodeSet[4]
    nodeSet[3].transition[(L,(GR,CL))] = nodeSet[2]
    nodeSet[3].transition[(L,(GR,SL))] = nodeSet[2]
    nodeSet[3].transition[(L,(GR,CR))] = nodeSet[2]

    nodeSet[2].transition[(L,(GL,CL))] = nodeSet[3]
    nodeSet[2].transition[(L,(GL,SL))] = nodeSet[3]
    nodeSet[2].transition[(L,(GL,CR))] = nodeSet[3]
    nodeSet[2].transition[(L,(GR,CL))] = nodeSet[1]
    nodeSet[2].transition[(L,(GR,SL))] = nodeSet[1]
    nodeSet[2].transition[(L,(GR,CR))] = nodeSet[1]

    nodeSet[4].transition[(L,(GL,CL))] = nodeSet[5]
    nodeSet[4].transition[(L,(GL,SL))] = nodeSet[5]
    nodeSet[4].transition[(L,(GL,CR))] = nodeSet[5]
    nodeSet[4].transition[(L,(GR,CL))] = nodeSet[3]
    nodeSet[4].transition[(L,(GR,SL))] = nodeSet[3]
    nodeSet[4].transition[(L,(GR,CR))] = nodeSet[3]

    nodeSet[1].transition[(OL,(GL,CL))] = nodeSet[3]
    nodeSet[1].transition[(OL,(GL,SL))] = nodeSet[3]
    nodeSet[1].transition[(OL,(GL,CR))] = nodeSet[3]
    nodeSet[1].transition[(OL,(GR,CL))] = nodeSet[3]
    nodeSet[1].transition[(OL,(GR,SL))] = nodeSet[3]
    nodeSet[1].transition[(OL,(GR,CR))] = nodeSet[3]

    nodeSet[5].transition[(OR,(GL,CL))] = nodeSet[3]
    nodeSet[5].transition[(OR,(GL,SL))] = nodeSet[3]
    nodeSet[5].transition[(OR,(GL,CR))] = nodeSet[3]
    nodeSet[5].transition[(OR,(GR,CL))] = nodeSet[3]
    nodeSet[5].transition[(OR,(GR,SL))] = nodeSet[3]
    nodeSet[5].transition[(OR,(GR,CR))] = nodeSet[3]

    return nodeSet
end

function initialize_fsm_frame_sets(pomdp::Union{Level_0_tigerPOMDP, Level_l_tigerPOMDP})
    tigerFSM = initialize_fsm(pomdp)
    tiger_FSM_frame_1 = FSM_Frame{Int,Tuple{Int,Int}}(1, tigerFSM)
    tiger_FSM_frame_2 = FSM_Frame{Int,Tuple{Int,Int}}(2, tigerFSM)
    tiger_sf_ag1 = Vector{Subintentional_Frame}(1)
    tiger_sf_ag1[1] = tiger_FSM_frame_1
    tiger_sf_ag2 = Vector{Subintentional_Frame}(1)
    tiger_sf_ag2[1] = tiger_FSM_frame_2
    agent_SF_sets = Vector{Vector{Subintentional_Frame}}(2)
    agent_SF_sets[1] = tiger_sf_ag1
    agent_SF_sets[2] = tiger_sf_ag2
    agent_SF_sets
end
