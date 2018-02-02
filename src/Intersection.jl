const CAR_LENGTH = 6.0
const CAR_WIDTH = 4.0

type CarPhysicalState2d
    state::NTuple{4,Float64}    #<x,y,θ,v>
end
==(a::CarPhysicalState2d, b::CarPhysicalState2d) = a.state==b.state
Base.hash(x::CarPhysicalState2d) = hash(x.state)

type CarAction2d
    accl::Float64
    ang_vel::Float64
end
==(a::CarAction2d, b::CarAction2d) = a.accl==b.accl && a.ang_vel==b.ang_vel
Base.hash(x::CarAction2d) = hash((x.accl,x.ang_vel))


type VehicleActionSpace_Intersection
    actions::Vector{CarAction2d}
end

VehicleActionSpace_Intersection() =
        VehicleActionSpace_Intersection([CarAction2d(-2.0, 0.0), CarAction2d(0.0, 0.0),
                CarAction2d(2.0, 0.0), CarAction2d(-6.0, 0.0)])
Base.length(asp::VehicleActionSpace_Intersection) = length(asp.actions)
iterator(actSpace::VehicleActionSpace_Intersection) = 1:length(actSpace.actions)
dimensions(::VehicleActionSpace_Intersection) = 1

#Sample random action
Base.rand(rng::AbstractRNG, asp::VehicleActionSpace_Intersection) = Base.rand(rng, 1:Base.length(asp))

function propagate_car(s::CarPhysicalState2d, a::CarAction2d, t::Float64, rng::AbstractRNG=Base.GLOBAL_RNG, noise::NTuple{4, Float64}=NTuple{4, Float64}((0.0,0.0,0.0,0.0)))
    x = s.state[1]
    y = s.state[2]
    θ = s.state[3]
    v = s.state[4]

    v_dot = a.accl
    ω = a.ang_vel

    xp = x + (v * cos(θ) * t + 0.5 * v_dot * cos(θ) * t^2) * (1 + randn(rng)*noise[1])
    yp = y + (v * sin(θ) * t + 0.5 * v_dot * sin(θ) * t^2) * (1 + randn(rng)*noise[2])
    θp = θ + (ω * t) * (1 + randn(rng)*noise[3])
    vp = v + (v_dot * t) * (1 + randn(rng)*noise[4])

    return CarPhysicalState2d((xp,yp,θp,vp))
end

type IntersectionState2d
    terminal::Int64
    agent_states::Tuple{CarPhysicalState2d, CarPhysicalState2d}
end

function IntersectionState2d(ag_st::Tuple{CarPhysicalState2d, CarPhysicalState2d})
    return IntersectionState2d(0,ag_st)
end
==(a::IntersectionState2d,b::IntersectionState2d) = a.terminal == b.terminal && a.agent_states == b.agent_states
Base.hash(x::IntersectionState2d) = hash((x.terminal,x.agent_states))


#NOTE: Temporary fix
function check_collision(s_i::CarPhysicalState2d, s_j::CarPhysicalState2d)
    x_i = s_i.state[1]
    y_i = s_i.state[2]

    x_j = s_j.state[1]
    y_j = s_j.state[2]

    if ((x_i-x_j)^2 + (y_i-y_j)^2 <= ((CAR_LENGTH + CAR_WIDTH)/2)^2)
        return true
    end
    return false
end

type IntersectionPOMDP <: POMDP{IntersectionState2d, Tuple{Int64,Int64}, Tuple{Int64,Int64}}
    desired_velocity::Float64
    collision_cost::Float64
    success_reward::Float64
    vel_dev_cost::Float64
    hard_brake_cost::Float64
    decision_timestep::Float64
    collision_check_timestep::Float64
    discount_factor::Float64
    agID::Int64
end

IntersectionPOMDP(;desired_velocity = 10.0, collision_cost = -500.0, success_reward = 100.0, vel_dev_cost = -2.0, hard_brake_cost = -10.0, decision_timestep = 1.0, collision_check_timestep = 0.25, discount_factor = 0.95, agID=1) = IntersectionPOMDP(desired_velocity, collision_cost, success_reward, vel_dev_cost, hard_brake_cost, decision_timestep, collision_check_timestep, discount_factor, agID)
==(a::IntersectionPOMDP, b::IntersectionPOMDP) = (a.agID == b.agID &&
        a.discount_factor == b.discount_factor && a.desired_velocity == b.desired_velocity &&
        a.collision_cost == b.collision_cost && a.success_reward == b.success_reward &&
        a.vel_dev_cost == b.vel_dev_cost && a.hard_brake_cost == b.hard_brake_cost)
Base.hash(a::IntersectionPOMDP) = hash((a.agID, a.discount_factor, a.desired_velocity,
                                        a.collision_cost,
                                            a.success_reward, a.vel_dev_cost, a.hard_brake_cost))
discount(p::IntersectionPOMDP) = p.discount_factor

function isterminal(p::IntersectionPOMDP, st::Tuple{CarPhysicalState2d,CarPhysicalState2d})
    if st.terminal != 0
        return true
    end
    return false
end

n_actions(p::IntersectionPOMDP) = length(actions(p))
function actions(::IntersectionPOMDP)
    caracts = VehicleActionSpace_Intersection()
    joint_actions = sizehint!(Vector{Tuple{Int64, Int64}}(), length(caracts.actions)^2)
    for i in 1:length(caracts.actions)
        for j in 1:length(caracts.actions)
            push!(joint_actions, (i,j))
        end
    end
    return joint_actions
end
function action_index(p::IntersectionPOMDP, a::Tuple{Int64, Int64})
    caracts = VehicleActionSpace_Intersection().actions
    i = (a[1]-1)*length(caracts) + a[2]
    return i
end
function observations(p::IntersectionPOMDP)
    observation_set = sizehint!(Vector{Tuple{Int64,Int64}}(), 100)
    for i_dist in 1:5
        for j_dist in 1:5

            push!(observation_set, (i_dist,j_dist))
        end
    end
    return observation_set
end

type IntersectionDistribution
    s1::CarPhysicalState2d
    s2::CarPhysicalState2d

    σ1::NTuple{4,Float64}
    σ2::NTuple{4,Float64}
end

function initial_state_distribution(p::IntersectionPOMDP)
    s1 = CarPhysicalState2d((-25.0, 0.0, 10.0, 0.0))
    s2 = CarPhysicalState2d((0.0, -25.0, 10.0, pi/2))
    σ1 = (0.0,0.0,0.0,0.0)
    σ2 = (0.0,0.0,0.0,0.0)
    return IntersectionDistribution(s1, s2, σ1, σ2)
end

function rand(rng, dist::IntersectionDistribution)
    s1 = (dist.s1.state[1] + randn(rng)*dist.σ1[1], dist.s1.state[2] + randn(rng)*dist.σ1[2],
            dist.s1.state[3] + randn(rng)*dist.σ1[3], dist.s1.state[4] + randn(rng)*dist.σ1[4])
    s2 = (dist.s2.state[1] + randn(rng)*dist.σ2[1], dist.s2.state[2] + randn(rng)*dist.σ2[2],
            dist.s2.state[3] + randn(rng)*dist.σ2[3], dist.s2.state[4] + randn(rng)*dist.σ2[4])
    return IntersectionState2d((CarPhysicalState2d(s1), CarPhysicalState2d(s2)))
end


function generate_s(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    if s.terminal >= 0
        return s
    end


    s_i = s.agent_states[1]
    s_j = s.agent_states[2]
    if check_collision(s_i, s_j)
        return IntersectionState2d(1, (s_i,s_j))
    end
    if s_i.state[1] >= 5.0 || s_j.state[2] >= 5.0
        return IntersectionState2d(2, (s_i, s_j))
    end

    caracts = VehicleActionSpace_Intersection().actions
    a_i = caracts[a[1]]
    a_j = caracts[a[2]]

    action_time_remain = p.decision_timestep
    while action_time_remain > 0.0
        action_duration = p.collision_check_timestep
        if action_duration > action_time_remain
            action_duration = action_time_remain
        end
        sp_i = propagate_car(s_i, a_i, action_duration, rng, (0.01,0.01,0.01,0.01))
        sp_j = propagate_car(s_j, a_j, action_duration, rng, (0.01,0.01,0.01,0.01))

        if check_collision(sp_i, sp_j)
            return IntersectionState2d(0, (sp_i,sp_j))
        end
    end

    return IntersectionState2d(0,(sp_i, sp_j))
end

function generate_o(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, sp::IntersectionState2d, rng::AbstractRNG)
    s_i = s.agent_states[1]
    s_j = s.agent_states[2]

    dist_i = abs(s_i.state[1])
    dist_j = abs(s_j.state[2])
    desired_velocity = p.desired_velocity

    o_i = 1
    if dist_i > 1 * desired_velocity && dist_i <= 2 * desired_velocity
        o_i = 2
    elseif dist_i > 2 * desired_velocity && dist_i <= 3 * desired_velocity
        o_i = 3
    elseif dist_i > 3 * desired_velocity && dist_i <= 5 * desired_velocity
        o_i = 4
    else
        o_i = 5
    end

    o_j = 1
    if dist_j > 1 * desired_velocity && dist_j <= 2 * desired_velocity
        o_j = 2
    elseif dist_j > 2 * desired_velocity && dist_j <= 3 * desired_velocity
        o_j = 3
    elseif dist_j > 3 * desired_velocity && dist_j <= 5 * desired_velocity
        o_j = 4
    else
        o_j = 5
    end

    return (o_i,o_j)
end

function reward(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    if s.terminal >= 0
        return 0.0
    end

    s_i = s.agent_states[1]
    s_j = s.agent_states[2]
    if check_collision(s_i, s_j)
        return p.collision_cost
    end
    caracts = VehicleActionSpace_Intersection().actions
    a_i = caracts[a[1]]
    a_j = caracts[a[2]]

    reward = 0.0
    #Agent-wise reward
    agID = p.agID
    if agID == 1
        if s_i.state[1] >= 5.0
            reward += p.success_reward
        end
        reward += abs(s_i.state[3] - p.desired_velocity) * p.vel_dev_cost
        if a_i.accl < -4.0
            reward += p.hard_brake_cost
        end
    else
        if s_j.state[2] >= 5.0
            reward += p.success_reward
        end
        reward += abs(s_j.state[3] - p.desired_velocity) * p.vel_dev_cost
        if a_j.accl < -4.0
            reward += p.hard_brake_cost
        end
    end

    return reward
end

function generate_sor(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    sp = generate_s(p,s,a,rng)
    o = generate_o(p,s,a,sp,rng)
    r = reward(p,s,a,rng)
    return sp,o,r
end

function generate_sr(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    r = reward(p, s, a, rng)
    return sp,r
end

function generate_so(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    o = generate_o(p, s, a, sp, rng)
    return sp,o
end

function generate_or(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, sp::IntersectionState2d, rng::AbstractRNG)
    o = generate_o(p,s,a,sp,rng)
    r = reward(p,s,a,rng)

    return o,r
end

function obs_weight(p::IntersectionPOMDP, s::IntersectionState2d, a::Tuple{Int64,Int64}, sp::IntersectionState2d, o::Tuple{Int64, Int64}, rng::AbstractRNG)
    o_generated = generate_o(p,s,a,sp,rng)
    if o == o_generated
        return 1.0
    end
    return 0.0
end

function initialize_static_distribution_frame_sets(::IntersectionPOMDP)
    #Starting with constant velocity model
    st_prob_dist_1 = Static_Distribution_Frame(1, Dict(1 => 0.0, 2 => 1.0, 3 => 0.0))
    st_prob_dist_2 = Static_Distribution_Frame(2, Dict(1 => 0.0, 2 => 1.0, 3 => 0.0))
    intersection_sf_1 = Vector{Subintentional_Frame}(1)
    intersection_sf_1[1] = st_prob_dist_1
    intersection_sf_2 = Vector{Subintentional_Frame}(1)
    intersection_sf_2[1] = st_prob_dist_2
    #push!(tiger_sm,st_prob_dist)
    agent_SF_sets = Vector{Vector{Subintentional_Frame}}(2)
    fill!(agent_SF_sets, Vector{Subintentional_Frame}())
    agent_SF_sets[1] = intersection_sf_1
    agent_SF_sets[2] = intersection_sf_2
    agent_SF_sets
end

function initialize_intentional_frame_sets(::IntersectionPOMDP)
    lvl_l_tiger_pomdp_1 = IntersectionPOMDP(agID=1)
    lvl_l_tiger_pomdp_2 = IntersectionPOMDP(agID=2)
    ag_intentional_frame_set_1 = Vector{POMDP}(1)
    ag_intentional_frame_set_1[1] = lvl_l_tiger_pomdp_1

    ag_intentional_frame_set_2 = Vector{POMDP}(1)
    ag_intentional_frame_set_2[1] = lvl_l_tiger_pomdp_2

    agent_frame_sets = [ag_intentional_frame_set_1, ag_intentional_frame_set_2]
end

function initial_intersection_ipomdp(lvl::Int64)   #Called for agent 1 only. the others are handled internally
    lvl_l_intersection_pomdp = IntersectionPOMDP()

    intersection_static_dist_frame_sets = initialize_static_distribution_frame_sets(lvl_l_intersection_pomdp)
    intersection_intentional_frame_sets = initialize_intentional_frame_sets(lvl_l_intersection_pomdp)

    return IPOMDP_2(1,lvl,lvl_l_intersection_pomdp, intersection_static_dist_frame_sets, intersection_intentional_frame_sets)
end

type Intersection_Frame_Distribution
    ipomdp::IPOMDP_2
    phy_state_distribution::IntersectionDistribution
    subintentional_cp::Vector{Float64}
    intentional_cp::Vector{Tuple{Float64, Intersection_Frame_Distribution}}
end

function Intersection_Frame_Distribution(ipomdp::IPOMDP_2, td::IntersectionDistribution)
    n_frames = length(ipomdp.oaSM) + length(ipomdp.oaFrames)    #Works for L0 as well
    sub_prob = sizehint!(Vector{Float64}(), length(ipomdp.oaSM))
    for i in 1:length(ipomdp.oaSM)
        push!(sub_prob, 1.0/n_frames)
    end
    int_prob = sizehint!(Vector{Tuple{Float64, Intersection_Frame_Distribution}}(), length(ipomdp.oaFrames))

    if level(ipomdp) > 0
        for i in 1:length(ipomdp.oaFrames)
            push!(int_prob, (1.0/n_frames, initial_state_distribution(ipomdp.oaFrames[i])))
        end
    end
    return Intersection_Frame_Distribution(ipomdp, td, sub_prob, int_prob)
end

#TODO: May not need to pass the pomdp if the distribution type is self identifying
function initial_ipomdp_frame_distribution(pomdp::IntersectionPOMDP, ipomdp::IPOMDP_2, intersection_dist::IntersectionDistribution)
    if !(typeof(ipomdp.thisPOMDP) <: IntersectionPOMDP)
        println("Not an intersection pomdp")
        return Vector{Float64}()
    end
    return Intersection_Frame_Distribution(ipomdp, intersection_dist)
end

function num_nested_particles(pomdp::IntersectionPOMDP, ipomdp::IPOMDP_2)
    num_particles = sizehint!(Vector{Int64}(),level(ipomdp)+1)
    if !(typeof(ipomdp.thisPOMDP) <: IntersectionPOMDP)
        println("Not an intersection pomdp")
        return num_particles
    end
    level(ipomdp) == 1 ? num_particles = [30,100] : num_particles = [10,50,300]
    return num_particles
end

function rand(rng, dist::Intersection_Frame_Distribution, n_particles::Vector{Int64})
    ipomdp = dist.ipomdp
    lvl = level(ipomdp)
    n_particle = n_particles[lvl+1]
    particles = sizehint!(Vector{InteractiveState{IntersectionState2d}}(), n_particle)
    for p in 1:n_particle
        s = rand(rng, dist.phy_state_distribution)
        #Sample frame
        frameIdx = 0
        rnd = rand(rng)
        sum = 0.0
        for frIdx in 1:length(dist.subintentional_cp)
            sum += dist.subintentional_cp[frIdx]
            if sum >= rnd
                frameIdx = frIdx
                break
            end
        end
        if rnd > sum
            for frIdx in 1:length(dist.intentional_cp)
                sum += dist.intentional_cp[frIdx][1]
                if sum >= rnd
                    frameIdx = frIdx + length(dist.subintentional_cp)
                    break
                end
            end
        end
        local frame::Frame
        local model::Model
        if frameIdx <= length(dist.subintentional_cp)
            #Subintentional
            frame = ipomdp.oaSM[frameIdx]
            model = sample_model(frame, rng)
        else
            #Intentional
            scaled_frameIdx = frameIdx-length(ipomdp.oaSM)
            frame = ipomdp.oaFrames[scaled_frameIdx]
            model = Intentional_Model(frame, rand(rng, dist.intentional_cp[scaled_frameIdx][2], n_particles))
        end
        push!(particles, InteractiveState{IntersectionState2d}(s,model))
    end

    return  InteractiveParticleCollection(particles)
end
