type IPOMDP_2{S,A1,A2,O1} <: AbstractIPOMDP
    #ag_ID::Int64
    level::Int64
    thisPOMDP::POMDP{S,Tuple{A1,A2},O1}
    oaSM::Vector{Subintentional_Frame}
    oaFrames::Vector{IPOMDP_2}
end

function IPOMDP_2(agID::Int64, level::Int64, agentPOMDP::POMDP,
        agentSF_sets::Vector{Vector{Subintentional_Frame}},
        agentFrameSets::Vector{Vector{POMDP}}
    )

    S = state_type(agentPOMDP)
    A = action_type(agentPOMDP)
    O = obs_type(agentPOMDP)
    A1 = A.parameters[1]
    A2 = A.parameters[2]
    agID == 1 ? oaID = 2 : oaID = 1
    oaSF = agentSF_sets[oaID]
    agentPOMDP.agID = agID  #Added here so that it gives a error if POMDP doesn't have a field agID

    oaFrames1 = Vector{IPOMDP_2}()
    if level > 0
        oaFrames = Vector{IPOMDP_2}()
        #println("Level = ", level)
        for oaPOMDP in agentFrameSets[oaID]
            for o_level in 0:level-1
                #println("O_level for $oaID = ", o_level)
                #println("Recursive call begin")
                #IPOMDP_2a(oaID, o_level, oaPOMDP, agentSF_sets, agentFrameSets)
                push!(oaFrames, IPOMDP_2(oaID, o_level, oaPOMDP, agentSF_sets, agentFrameSets))
                #println("Recursive call end")
            end
        end
        oaFrames1 = oaFrames
    end
    return IPOMDP_2{S,A1,A2,O}(level, agentPOMDP, oaSF, oaFrames1)
end
==(a::IPOMDP_2, b::IPOMDP_2) = a.level==b.level && a.thisPOMDP == b.thisPOMDP && a.oaSM == b.oaSM && a.oaFrames == b.oaFrames
Base.hash(a::IPOMDP_2) = hash((a.level,a.thisPOMDP,a.oaSM,a.oaFrames))

function sparse_print(ipomdp::IPOMDP_2)
    print("AgID:$(agentID(ipomdp)) Lvl:$(ipomdp.level)")
end

discount(ip::IPOMDP_2) = discount(ip.thisPOMDP)
agentID(ip::IPOMDP_2) = ip.thisPOMDP.agID
level(ip::IPOMDP_2) = ip.level

state_type{S,A1,A2,O1}(ipomdp::IPOMDP_2{S,A1,A2,O1}) = S
action_type{S,A1,A2,O1}(ipomdp::IPOMDP_2{S,A1,A2,O1}) = A1
oaction_type{S,A1,A2,O1}(ipomdp::IPOMDP_2{S,A1,A2,O1}) = A2
obs_type{S,A1,A2,O1}(ipomdp::IPOMDP_2{S,A1,A2,O1}) = O1

function actions(ipomdp::IPOMDP_2)
    agID = agentID(ipomdp)
    A = action_type(ipomdp)
    actvec = Vector{A}()
    for a in actions(ipomdp.thisPOMDP)
        push!(actvec, a[agID])
    end
    actset =  Set(actvec)
    acts = sizehint!(Vector{A}(), length(actset))
    for a in actset
        push!(acts, a)
    end
    return acts
end

function oactions(ipomdp::IPOMDP_2)
    agID = agentID(ipomdp)
    oaID = 0
    agID == 1 ? oaID = 2 : oaID = 1
    A = oaction_type(ipomdp)
    actvec = Vector{A}()
    for a in actions(ipomdp.thisPOMDP)
        push!(actvec, a[oaID])
    end
    actset = Set(actvec)
    acts = sizehint!(Vector{A}(), length(actset))
    for a in actset
        push!(acts, a)
    end
    return acts
end

n_actions(ipomdp::IPOMDP_2) = length(actions(ipomdp))
n_oactions(ipomdp::IPOMDP_2) = length(oactions(ipomdp))

function joint_actions(ipomdp::IPOMDP_2)
    return actions(ipomdp.thisPOMDP)
end

function observations(ipomdp::IPOMDP_2)
    return observations(ipomdp.thisPOMDP)
end

function generate_s{S,A1,A2}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end
    return generate_s(ipomdp.thisPOMDP, s, (a1,a2), rng, Nullable(frame_j))
end

function generate_o{S,A1,A2}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, sp::S, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end
    generate_o(ipomdp.thisPOMDP, s, (a1,a2), sp, rng, Nullable(frame_j))
end

function reward{S,A1,A2}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end

    return reward(ipomdp.thisPOMDP, s, (a1,a2), rng, Nullable(frame_j))
end

function generate_sor{S,A1,A2}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end
    #println("a1 = $a1, a2 = $a2")
    return generate_sor(ipomdp.thisPOMDP, s, (a1,a2), rng, Nullable(frame_j))
end

function generate_sr{S,A1,A2}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end

    return generate_sr(ipomdp.thisPOMDP, s, (a1,a2), rng, Nullable(frame_j))
end

function generate_so{S,A1,A2}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end

    return generate_so(ipomdp.thisPOMDP, s, (a1,a2), rng, Nullable(frame_j))
end

function generate_or{S,A1,A2}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, sp::S, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end

    return generate_or(ipomdp.thisPOMDP, s, (a1,a2), sp, rng, Nullable(frame_j))
end

function obs_weight{S,A1,A2,O}(ipomdp::IPOMDP_2, s::S, a::A1, oa::A2, sp::S, o::O, frame_j::Frame,
                    rng::AbstractRNG=Base.GLOBAL_RNG)
    if agentID(ipomdp) == 1
        a1 = a
        a2 = oa
    else
        a1 = oa
        a2 = a
    end
    return obs_weight(ipomdp.thisPOMDP, s, (a1,a2), sp, o, rng, Nullable(frame_j))
end

function initial_state_distribution(ipomdp::IPOMDP_2)
    return initial_ipomdp_frame_distribution(ipomdp.thisPOMDP, ipomdp, initial_state_distribution(ipomdp.thisPOMDP))
end
