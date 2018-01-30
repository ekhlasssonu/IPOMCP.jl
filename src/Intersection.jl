const CAR_LENGTH = 6.0
const CAR_WIDTH = 4.0

type CarPhysicalState2d
    state::NTuple{4,Float64}    #<x,y,θ,v>
end

type CarAction2d
    accl::Float64
    ang_vel::Float64
end

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

    ̇v = a.accl
    ω = a.ang_vel

    xp = x + (v * cos(θ) * t + 0.5 * ̇v * cos(θ) * t^2) * (1 + randn(rng)*noise[1])
    yp = y + (v * sin(θ) * t + 0.5 * ̇v * sin(θ) * t^2) * (1 + randn(rng)*noise[2])
    θp = θ + (ω * t) * (1 + randn(rng)*noise[3])
    vp = v + ̇(v * t) * (1 + randn(rng)*noise[4])

    return CarPhysicalState2d((xp,yp,θp,vp))
end

type IntersectionState2d
    terminal::Int64
    agent_states::Tuple{CarPhysicalState2d, CarPhysicalState2d}
end

IntersectionState2d(agst::Tuple{CarPhysicalState2d, CarPhysicalState2d})
                = IntersectionState2d(0,agSt)

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

IntersectionPOMDP() = IntersectionPOMDP(10.0, -500.0, 100.0, -2.0, -10.0, 1.0, 0.25, 0.95, 1)

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
    for i_dist in 1:10
        for j_dist in 1:10

            push!(observation_set, (i,j))
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
