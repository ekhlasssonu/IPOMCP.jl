#Observations
const GL = 1
const GR = 2
const CL = 1
const SL = 2
const CR = 3

#Actions
const OL = 1
const L  = 2
const OR = 3

#States
const TL = 1
const TR = 2


type Level_l_tigerPOMDP <: POMDP{Int64, Tuple{Int64,Int64}, Tuple{Int64,Int64}}
    r_listen::Float64 # reward for listening (default -1)
    r_findtiger::Float64 # reward for finding the tiger (default -100)
    r_findgold::Float64 # reward for escaping (default 10)
    p_growl::Float64 # prob of correctly listening a growl(default 0.85)
    p_creak::Float64 # prob of correctly listening a creak(default 0.9)
    p_reset::Float64 # prob of tiger switching door on opening the door
    discount_factor::Float64 # discount

    agID::Int64
end

Level_l_tigerPOMDP(;r_listen=-1.0, r_findtiger=-100.0, r_findgold=10.0, p_growl=0.85, p_creak=0.9, p_reset=0.5, discount=0.95, agID=1) = Level_l_tigerPOMDP(r_listen, r_findtiger, r_findgold, p_growl, p_reset, p_creak, discount,agID)

discount(p::Level_l_tigerPOMDP) = p.discount_factor

function isterminal(p::Level_l_tigerPOMDP, st::Int64)
  false
end

n_states(p::Level_l_tigerPOMDP) = 2
n_actions(p::Level_l_tigerPOMDP) = length(actions(p))
actions(::Level_l_tigerPOMDP) = [(OL,OL), (OL,L), (OL,OR), (L,OL), (L,L), (L,OR), (OR,OL), (OR,L), (OR,OR)]
action_index(::Level_l_tigerPOMDP, a::Tuple{Int64, Int64}) = (a[1]-1)*3 + a[2]
observations(::Level_l_tigerPOMDP) = [(GL, CL),(GL, SL),(GL, CR),(GR, CL),(GR, SL),(GR, CR)]

type TigerDistribution
    p::Float64
    #it::Vector{Int64}
end
TigerDistribution() = TigerDistribution(0.5)

function initial_state_distribution(p::Level_l_tigerPOMDP)
    return TigerDistribution()
end
function rand(rng, dist::TigerDistribution)
    s = 0
    rand(rng) < dist.p ? s=1 : s=2
    return s
end
#=function initial_state(p::Level_l_tigerPOMDP, rng::AbstractRNG=MersenneTwister(1))
    rand(rng, initial_state_distribution(p))
end=#

function generate_s(p::Level_l_tigerPOMDP, s::Int64, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    #generate s
    sp = s
    #If either agent opens, the tiger stays behind its door with prob 1-p.p_reset or switches
    if a[1]!=L || a[2]!=L
        if rand(rng) < p.p_reset
            rand(rng) < 0.5 ? sp = TR : sp = TL
        end
    end
end
function generate_o(p::Level_l_tigerPOMDP, s::Int64, a::Tuple{Int64,Int64}, sp::Int64, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end
    growl = GL
    creak = CL

    if a[agID] == OL || a[agID] == OR
        rand(rng) < 0.5 ? growl=GL : growl=GR
    else
        if rand(rng) < p.p_growl
            sp == TL ? growl = GL : growl = GR
        else
            sp == TL ? growl = GR : growl = GL
        end
    end
    if a[agID] == OL || a[agID] == OR
        rnd = rand(rng)
        rnd < 0.1 ? creak=CL : rnd < 0.9 ? creak=SL : creak=CR
    else
        rnd = rand(rng)
        if rnd < p.p_creak
            if a[oaID] == OL
                creak = CL
            elseif a[oaID] == L
                creak = SL
            else
                creak = CR
            end
        else
            if a[oaID] == OL
                rnd < p.p_creak + (1-p.p_creak)/2.0 ? creak = CR : creak = SL
            elseif a[oaID] == L
                rnd < p.p_creak + (1-p.p_creak)/2.0 ? creak = CR : creak = CL
            else
                rnd < p.p_creak + (1-p.p_creak)/2.0 ? creak = CL : creak = SL
            end
        end
    end
    o = (growl, creak)

    return o
end

function generate_sor(p::Level_l_tigerPOMDP, s::Int64, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    #generate s
    sp = s
    #If either agent opens, the tiger stays behind its door with prob 1-p.p_reset or switches
    if a[1]!=L || a[2]!=L
        if rand(rng) < p.p_reset
            rand(rng) < 0.5 ? sp = TR : sp = TL
        end
    end

    #generate o
    growl = GL
    creak = CL

    if a[agID] == OL || a[agID] == OR
        rand(rng) < 0.5 ? growl=GL : growl=GR
    else
        if rand(rng) < p.p_growl
            sp == TL ? growl = GL : growl = GR
        else
            sp == TL ? growl = GR : growl = GL
        end
    end
    if a[agID] == OL || a[agID] == OR
        rnd = rand(rng)
        rnd < 0.1 ? creak=CL : rnd < 0.9 ? creak=SL : creak=CR
    else
        rnd = rand(rng)
        if rnd < p.p_creak
            if a[oaID] == OL
                creak = CL
            elseif a[oaID] == L
                creak = SL
            else
                creak = CR
            end
        else
            if a[oaID] == OL
                rnd < p.p_creak + (1-p.p_creak)/2.0 ? creak = CR : creak = SL
            elseif a[oaID] == L
                rnd < p.p_creak + (1-p.p_creak)/2.0 ? creak = CR : creak = CL
            else
                rnd < p.p_creak + (1-p.p_creak)/2.0 ? creak = CL : creak = SL
            end
        end
    end
    o = (growl, creak)

    #generate r
    r = 0
    if a[agID] == L
        r = p.r_listen
    end
    if s == TL
        if a[agID] == OL
            r = p.r_findtiger
        elseif a[agID] == OR
            r = p.r_findgold
        end
    end
    if s == TR
        if a[agID] == OL
            r = p.r_findgold
        elseif a[agID] == OR
            r = p.r_findtiger
        end
    end
    return s,o,r
end
function generate_so(p::Level_l_tigerPOMDP, s::Int64, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    o = generate_o(p, s, a, sp, rng)
    return sp,o
end
function reward(p::Level_l_tigerPOMDP, s::Int64, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end
    #generate r
    r = 0
    if a[agID] == L
        r = p.r_listen
    end
    if s == TL
        if a[agID] == OL
            r = p.r_findtiger
        elseif a[agID] == OR
            r = p.r_findgold
        end
    end
    if s == TR
        if a[agID] == OL
            r = p.r_findgold
        elseif a[agID] == OR
            r = p.r_findtiger
        end
    end
    return r
end

function generate_or(p::Level_l_tigerPOMDP, s::Int64, a::Tuple{Int64,Int64}, sp::Int64, rng::AbstractRNG)
    o = generate_o(p,s,a,sp,rng)
    r = reward(p,s,a,rng)

    return o,r
end

function obs_weight(p::Level_l_tigerPOMDP, s::Int64, a::Tuple{Int64,Int64}, sp::Int64, o::Tuple{Int64, Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    a_i = a[agID]
    a_j = a[oaID]

    o_prob = 1.0
    if a_i != L
        o_prob *= 0.5
        if o[2] == CL
            o_prob *= 0.1
        elseif o[2] == SL
            o_prob *= 0.8
        else
            o_prob *= 0.1
        end
    else
        sp == TL ? (o[1] == GL ? o_prob *= p.p_growl : o_prob *= (1-p.p_growl)) : (o[1] == GR ? o_prob *= p.p_growl : o_prob *= (1-p.p_growl))
        if a_j == OL
            if o[2] == CL
                o_prob *= p.p_creak
            else
                o_prob *= (1-p.p_creak)/2.0
            end
        elseif a_j == L
            if o[2] == SL
                o_prob *= p.p_creak
            else
                o_prob *= (1-p.p_creak)/2.0
            end
        else
            if o[2] == CR
                o_prob *= p.p_creak
            else
                o_prob *= (1-p.p_creak)/2.0
            end
        end
    end

    return o_prob
end

function initialize_static_distribution_frame_sets(::Level_l_tigerPOMDP)
    st_prob_dist_1 = Static_Distribution_Frame(1, Dict(OL => 0.1, L => 0.8, OR => 0.1))
    st_prob_dist_2 = Static_Distribution_Frame(2, Dict(OL => 0.1, L => 0.8, OR => 0.1))
    tiger_sf_1 = Vector{Subintentional_Frame}(1)
    tiger_sf_1[1] = st_prob_dist_1
    tiger_sf_2 = Vector{Subintentional_Frame}(1)
    tiger_sf_2[1] = st_prob_dist_2
    #push!(tiger_sm,st_prob_dist)
    agent_SF_sets = Vector{Vector{Subintentional_Frame}}(2)
    fill!(agent_SF_sets, Vector{Subintentional_Frame}())
    agent_SF_sets[1] = tiger_sf_1
    agent_SF_sets[2] = tiger_sf_2
    agent_SF_sets
end

function initialize_intentional_frame_sets(::Level_l_tigerPOMDP)
    lvl_l_tiger_pomdp_1 = Level_l_tigerPOMDP(agID=1)
    lvl_l_tiger_pomdp_2 = Level_l_tigerPOMDP(agID=2)
    ag_intentional_frame_set = Vector{POMDP}(1)
    ag_intentional_frame_set[1] = lvl_l_tiger_pomdp_1

    oa_intentional_frame_set = Vector{POMDP}(1)
    oa_intentional_frame_set[1] = lvl_l_tiger_pomdp_2

    agent_frame_sets = [ag_intentional_frame_set, oa_intentional_frame_set]
end


function initial_tiger_ipomdp(lvl::Int64)   #Called for agent 1 only. the others are handled internally
    lvl_l_tiger_pomdp = Level_l_tigerPOMDP()

    tiger_static_dist_frame_sets = initialize_static_distribution_frame_sets(lvl_l_tiger_pomdp)
    tiger_intentional_frame_sets = initialize_intentional_frame_sets(lvl_l_tiger_pomdp)

    return IPOMDP_2(1,lvl,lvl_l_tiger_pomdp, tiger_static_dist_frame_sets, tiger_intentional_frame_sets)
end

type Tiger_Frame_Distribution
    ipomdp::IPOMDP_2
    p::Float64  #Distribution over physical state
    subintentional_cp::Vector{Vector{Float64}}
    intentional_cp::Vector{Vector{Tuple{Float64, Tiger_Frame_Distribution}}}
end

function Tiger_Frame_Distribution(ipomdp::IPOMDP_2, td::TigerDistribution)
    p = td.p
    n_frames = length(ipomdp.oaSM) + length(ipomdp.oaFrames)    #Works for L0 as well
    subintentional_cp = sizehint!(Vector{Vector{Float64}}(),2)
    sub_prob = sizehint!(Vector{Float64}(), length(ipomdp.oaSM))
    for i in 1:length(ipomdp.oaSM)
        push!(sub_prob, 1.0/n_frames)
    end
    for i in 1:2    # for all state
        push!(subintentional_cp, sub_prob)
    end
    if level(ipomdp) > 0
        intentional_cp = sizehint!(Vector{Vector{Tuple{Float64, Tiger_Frame_Distribution}}}(),2)
        int_prob = sizehint!(Vector{Tuple{Float64, Tiger_Frame_Distribution}}(), length(ipomdp.oaFrames))
        for i in 1:length(ipomdp.oaFrames)
            push!(int_prob, (1.0/n_frames, initial_state_distribution(ipomdp.oaFrames[i])))
        end
        for i in 1:2
            push!(intentional_cp, int_prob)
        end
    else
        intentional_cp = sizehint!(Vector{Vector{Tuple{Float64, Tiger_Frame_Distribution}}}(),0)
    end
    return Tiger_Frame_Distribution(ipomdp, p, subintentional_cp, intentional_cp)
end

#TODO: May not need to pass the pomdp if the distribution type is self identifying
function initial_ipomdp_frame_distribution(pomdp::Level_l_tigerPOMDP, ipomdp::IPOMDP_2, td::TigerDistribution)
    if !(typeof(ipomdp.thisPOMDP) <: Level_l_tigerPOMDP)
        println("Not a tiger pomdp")
        return Vector{Float64}()
    end
    return Tiger_Frame_Distribution(ipomdp, td)
end

function num_nested_particles(pomdp::Level_l_tigerPOMDP, ipomdp::IPOMDP_2)
    num_particles = sizehint!(Vector{Int64}(),level(ipomdp)+1)
    if !(typeof(ipomdp.thisPOMDP) <: Level_l_tigerPOMDP)
        println("Not a tiger pomdp")
        return num_particles
    end
    level(ipomdp) == 1 ? num_particles = [30,100] : num_particles = [10,50,300]
    return num_particles
end

#Nested belief returned
function rand(rng, dist::Tiger_Frame_Distribution, n_particles::Vector{Int64})
    ipomdp = dist.ipomdp
    lvl = level(ipomdp)
    n_particle = n_particles[lvl+1]
    particles = sizehint!(Vector{InteractiveState}(), n_particle)
    for p in 1:n_particle
        local s::Int64
        rand(rng) < dist.p ? s=1 : s=2
        #Sample frame
        frameIdx = 0
        rnd = rand(rng)
        sum = 0.0
        for frIdx in 1:length(dist.subintentional_cp[s])
            sum += dist.subintentional_cp[s][frIdx]
            if sum >= rnd
                frameIdx = frIdx
                break
            end
        end
        if rnd > sum
            for frIdx in 1:length(dist.intentional_cp[s])
                sum += dist.intentional_cp[s][frIdx][1]
                if sum >= rnd
                    frameIdx = frIdx + length(dist.subintentional_cp[s])
                    break
                end
            end
        end
        local frame::Frame
        local model::Model
        if frameIdx <= length(dist.subintentional_cp[s])
            #Subintentional
            frame = ipomdp.oaSM[frameIdx]
            model = sample_model(frame, rng)
        else
            #Intentional
            scaled_frameIdx = frameIdx-length(ipomdp.oaSM)
            frame = ipomdp.oaFrames[scaled_frameIdx]
            model = Intentional_Model(frame, rand(rng, dist.intentional_cp[s][scaled_frameIdx][2], n_particles))
        end
        push!(particles, InteractiveState{Int64}(s,model))
    end

    return InteractiveParticleCollection{InteractiveState{Int64}}(ParticleCollection{InteractiveState{Int64}}(particles))
end
