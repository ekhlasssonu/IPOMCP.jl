const CROSSWALK_WIDTH = 4.0

type PedestrianState2d
    x::Float64
    y::Float64
end
==(a::PedestrianState2d, b::PedestrianState2d) = a.x==b.x && a.y == b.y
function Base.hash(s::PedestrianState2d, h::UInt64=zero(UInt64))
    #print("*")
    return hash((s.x, s.y),h)
end
PedestrianState2d(s::Tuple{Float64,Float64}) = PedestrianState2d(s[1],s[2])

function print(s::PedestrianState2d)
    print("(",s.x,",",s.y,")")
end

#CarAction2d defined in Intersection.jl
type VehicleActionSpace_PedXing
    actions::Vector{CarAction2d}
end
VehicleActionSpace_PedXing() =
        VehicleActionSpace_PedXing([CarAction2d(-3.0, 0.0), CarAction2d(-2.0, 0.0),
                                    CarAction2d(-1.0, 0.0), CarAction2d(0.0, 0.0),
                                    CarAction2d(1.0, 0.0), CarAction2d(2.0, 0.0),
                                    CarAction2d(3.0, 0.0), CarAction2d(-6.0, 0.0)])
Base.length(asp::VehicleActionSpace_PedXing) = length(asp.actions)
iterator(actSpace::VehicleActionSpace_PedXing) = 1:length(actSpace.actions)
dimensions(::VehicleActionSpace_PedXing) = 1
Base.rand(rng::AbstractRNG, asp::VehicleActionSpace_PedXing) = Base.rand(rng, 1:Base.length(asp))

type PedAction2d
    v::Float64
end
==(a::PedAction2d, b::PedAction2d) = a.v == b.v
Base.hash(x::PedAction2d, h::UInt64=zero(UInt64)) = hash(x.v,h)

type PedActionSpace_PedXing
    actions::Vector{PedAction2d}
end
PedActionSpace_PedXing() = PedActionSpace_PedXing([PedAction2d(0.0), PedAction2d(0.6),
                    PedAction2d(1.0), PedAction2d(1.4), PedAction2d(1.8), PedAction2d(2.0)])
Base.length(asp::PedActionSpace_PedXing) = length(asp.actions)
iterator(actSpace::PedActionSpace_PedXing) = 1:length(actSpace.actions)
dimensions(::PedActionSpace_PedXing) = 1
Base.rand(rng::AbstractRNG, asp::PedActionSpace_PedXing) = Base.rand(rng, 1:Base.length(asp))

function propagate_ped(s::PedestrianState2d, a::PedAction2d, t::Float64, rng::AbstractRNG=Base.GLOBAL_RNG, noise::NTuple{2, Float64}=NTuple{2, Float64}((0.0,0.0)))
    x = s.x
    y = s.y
    v = a.v

    xp = x + randn(rng)*noise[1]
    yp = y + v*t *(1+randn(rng)*noise[2])

    return PedestrianState2d(xp,yp)
end

type PedXingState2d
    terminal::Int64
    agent_states::Tuple{CarPhysicalState2d, PedestrianState2d}
end

function PedXingState2d(ag_st::Tuple{CarPhysicalState2d, PedestrianState2d})
    return PedXingState2d(0,ag_st)
end
==(a::PedXingState2d,b::PedXingState2d) = a.terminal == b.terminal && a.agent_states == b.agent_states
function Base.hash(x::PedXingState2d, h::UInt64=zero(UInt64))
    return hash((x.terminal,x.agent_states),h)
end

function print(s::PedXingState2d)
    print("[",s.terminal,", ")
    print(s.agent_states[1])
    print(", ")
    print(s.agent_states[2])
    print("]")
end

function check_collision(s_i::CarPhysicalState2d, s_j::PedestrianState2d, buffer_dist::Float64=0.5, crosswalk_width::Float64 = CROSSWALK_WIDTH)
    x_i = s_i.state[1]
    y_i = s_i.state[2]

    x_j = s_j.x
    y_j = s_j.y

    if y_j < -crosswalk_width/2.0 || y_j > crosswalk_width/2.0
        return false
    end

    if (abs(x_i - x_j) <= CAR_LENGTH/2 + buffer_dist && abs(y_i - y_j) <= CAR_WIDTH/2 + buffer_dist)
        return true
    end
    return false
end

type PedXingPOMDP_Ped <: POMDP{PedXingState2d, Tuple{Int64,Int64}, Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}}
    desired_velocity::Float64       # 0.0, 1.0, 1.4, 1.8(, 0.6, 1.8)
    collision_cost::Float64         # Big negative for collision
    success_reward::Float64         # 0
    vel_dev_cost_coeff::Float64     # vel dev probably < 1.0
    vel_dev_cost_pow::Float64       # - (abs(desired_velocity - v) * coeff) ^ pow
    decision_timestep::Float64
    collision_check_timestep::Float64
    discount_factor::Float64
    agID::Int64
end

PedXingPOMDP_Ped(;desired_velocity = 1.0, collision_cost = -1000.0, success_reward = 0.0,
            vel_dev_cost_coeff = 5.0, vel_dev_cost_pow = 3.0, decision_timestep = 1.0,
            collision_check_timestep = 0.25, discount_factor = 0.95, agID=2) = PedXingPOMDP_Ped(
                desired_velocity, collision_cost, success_reward, vel_dev_cost_coeff, vel_dev_cost_pow,
                decision_timestep, collision_check_timestep, discount_factor, agID)

==(a::PedXingPOMDP_Ped, b::PedXingPOMDP_Ped) = (a.agID == b.agID &&
        a.discount_factor == b.discount_factor && a.desired_velocity == b.desired_velocity &&
        a.collision_cost == b.collision_cost && a.success_reward == b.success_reward &&
        a.vel_dev_cost_coeff == b.vel_dev_cost_coeff && a.vel_dev_cost_pow == b.vel_dev_cost_pow)
Base.hash(a::PedXingPOMDP_Ped, h::UInt64=zero(UInt64)) = hash((a.agID, a.discount_factor,
                                                            a.desired_velocity,a.collision_cost,
                                                            a.success_reward, a.vel_dev_cost_coeff,
                                                            a.vel_dev_cost_pow),h)



# Vehicle POMDP

type PedXingPOMDP_Veh <: POMDP{PedXingState2d, Tuple{Int64,Int64}, Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}}
    desired_velocity::Float64       # 10 m/s
    collision_cost::Float64         # Big negative for collision
    success_reward::Float64         # 0
    veh_vel_dev_cost_coeff::Float64
    ped_vel_dev_cost_coeff::Float64     # ped_vel_dev probably < 1.0
    ped_vel_dev_cost_pow::Float64       # - (abs(ped_desired_velocity - v) * coeff) ^ pow for intimidation
    hard_brake_cost::Float64
    decision_timestep::Float64
    collision_check_timestep::Float64
    discount_factor::Float64
    agID::Int64
end

PedXingPOMDP_Veh(;desired_velocity = 10.0, collision_cost = -1000.0, success_reward = 0.0, veh_vel_dev_cost_coeff = 1.0,
            ped_vel_dev_cost_coeff = 5.0, ped_vel_dev_cost_pow = 3.0, hard_brake_cost = -5.0,
            decision_timestep = 1.0, collision_check_timestep = 0.25, discount_factor = 0.95, agID=1) = PedXingPOMDP_Veh(
                desired_velocity, collision_cost, success_reward, veh_vel_dev_cost_coeff, ped_vel_dev_cost_coeff,
                ped_vel_dev_cost_pow, hard_brake_cost, decision_timestep, collision_check_timestep, discount_factor, agID)

==(a::PedXingPOMDP_Veh, b::PedXingPOMDP_Veh) = (a.agID == b.agID &&
        a.discount_factor == b.discount_factor && a.desired_velocity == b.desired_velocity &&
        a.collision_cost == b.collision_cost && a.success_reward == b.success_reward &&
        a.veh_vel_dev_cost_coeff == b. veh_vel_dev_cost_coeff && a.ped_vel_dev_cost_coeff == b.ped_vel_dev_cost_coeff &&
        a.hard_brake_cost == b.hard_brake_cost && a.ped_vel_dev_cost_pow == b.ped_vel_dev_cost_pow)
Base.hash(a::PedXingPOMDP_Veh, h::UInt64=zero(UInt64)) = hash((a.agID, a.discount_factor,
                                                            a.desired_velocity,a.collision_cost,
                                                            a.success_reward, a.veh_vel_dev_cost_coeff,
                                                            a.ped_vel_dev_cost_coeff,
                                                            a.ped_vel_dev_cost_pow, a.hard_brake_cost),h)

discount(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}) = p.discount_factor

function isterminal(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, st::PedXingState2d)
    if st.terminal > 0
        return true
    end
    return false
end

n_actions(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}) = (length(VehicleActionSpace_PedXing().actions) *
                                    length(PedActionSpace_PedXing().actions))
function actions(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped})
    caracts = VehicleActionSpace_PedXing()
    pedacts = PedActionSpace_PedXing()
    joint_actions = sizehint!(Vector{Tuple{Int64, Int64}}(), n_actions(p))
    for i in 1:length(caracts.actions)
        for j in 1:length(pedacts.actions)
            push!(joint_actions, (i,j))
        end
    end
    return joint_actions
end
function action_index(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, a::Tuple{Int64, Int64})
    caracts = VehicleActionSpace_PedXing()
    pedacts = PedActionSpace_PedXing()
    if a[1] < 1 || a[2] < 1 || a[1] > length(caracts.actions) || a[2] > length(pedacts.actions)
        return 0
    end
    i = (a[1]-1)*length(pedacts.actions) + a[2]
    return i
end
function observations(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped})
    observation_set = sizehint!(Vector{Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}}(), 100)
    for i_dist in 1:5   #Vehicle Positions  (0.0m<x, -5.0m<x≦0.0m, -10.0m<x≦-5.0m, -20.0m<x≦-10.0m, x≦-20.0 m )
        for i_vel in 1:4    #Vehicle Velocities (v≦1.0 m/s, 1.0 m/s<v≦5.0 m/s, 5.0m/s<v≦10.0m/s, 10.0m/s<v)
            for j_dist in 1:12   #Ped Positions (y≦-2.0m, -2.0m<y≦-1.0m, -1.0m<y≦0.0m, 0.0m<y≦1.0m, 1.0m<y≦2.0m, 2.0m<y)
                for j_vel in 1:5    #Ped Velocities (v≦0.5m/s, 0.5m/s<v≦0.9m/s, 0.9m/s<v≦1.3m/s, 1.3 m/s<v≦1.7m/s, 1.7m/s<v)
                    push!(observation_set, ((i_dist,i_vel),(j_dist,j_vel)))
                end
            end
        end
    end
    return observation_set
end

type PedXingDistribution
    s1::CarPhysicalState2d
    s2::PedestrianState2d

    σ1::NTuple{4,Float64}
    σ2::NTuple{2,Float64}
end

function initial_state_distribution(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped})
    s1 = CarPhysicalState2d((-35.0, 0.0, 0.0, 10.0))
    s2 = PedestrianState2d(0.0, -CROSSWALK_WIDTH/2.0 - 0.5)
    σ1 = (0.0,0.0,0.0,0.0)
    σ2 = (0.0,0.0)
    return PedXingDistribution(s1, s2, σ1, σ2)
end

function reset_state_distribution(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped},
                        s1::CarPhysicalState2d, s2::PedestrianState2d,
                        u1::NTuple{4,Float64}, u2::NTuple{2,Float64})
    return PedXingDistribution(s1,s2,u1,u2)
end

function rand(rng, dist::PedXingDistribution)
    s1 = (dist.s1.state[1] + randn(rng)*dist.σ1[1], dist.s1.state[2] + randn(rng)*dist.σ1[2],
            dist.s1.state[3] + randn(rng)*dist.σ1[3], dist.s1.state[4] + randn(rng)*dist.σ1[4])
    s2 = (dist.s2.x + randn(rng)*dist.σ2[1], dist.s2.y + randn(rng)*dist.σ2[2])
    return PedXingState2d((CarPhysicalState2d(s1), PedestrianState2d(s2)))
end

function generate_s(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, s::PedXingState2d,
                        a::Tuple{Int64,Int64}, rng::AbstractRNG, frame_j::Nullable = Nullable())
    if s.terminal > 0
        return s
    end
    s_i = s.agent_states[1]
    s_j = s.agent_states[2]
    if check_collision(s_i, s_j)
        #println("\t\tCollision")
        return PedXingState2d(1, (s_i,s_j))
    end
    if s_i.state[1] >= 5.0 && s_j.y >= CROSSWALK_WIDTH/2.0 + 0.5
        #println("\t\tSuccess")
        return PedXingState2d(2, (s_i, s_j))
    end

    caracts = VehicleActionSpace_PedXing().actions
    pedacts = PedActionSpace_PedXing().actions
    a_i = caracts[a[1]]
    a_j = pedacts[a[2]]

    #println("\t\ta1 = $(a[1]), a2 = $(a[2])")
    action_time_remain = p.decision_timestep
    #println("action time = ", action_time_remain, " a_i = $a_i, a_j = $a_j")
    sp_i = CarPhysicalState2d(s_i.state)
    sp_j = PedestrianState2d(s_j.x, s_j.y)
    while action_time_remain > 0.0
        action_duration = p.collision_check_timestep
        #println("Time remaining ",action_time_remain, " duration: ",action_duration)
        if action_duration > action_time_remain
            action_duration = action_time_remain
        end
        sp_i = propagate_car(sp_i, a_i, action_duration, rng, (0.01,0.01,0.01,0.01))
        sp_j = propagate_ped(sp_j, a_j, action_duration, rng, (0.05,0.02))

        action_time_remain -= action_duration

        if check_collision(sp_i, sp_j)
            return PedXingState2d(1,  (sp_i,sp_j))
        end
    end
    #println("s_i = ", s_i.state, " sp_i = ",sp_i.state)
    #println("s_j = ", s_j.state, " sp_j = ",sp_j.state)
    #NOTE: Initial state
    if sp_i.state[1] >= 5.0 && sp_j.y >= CROSSWALK_WIDTH/2.0 + 0.5
        return PedXingState2d(2, (sp_i, sp_j))
    end
    return PedXingState2d(0, (sp_i, sp_j))
end

function generate_o(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, s::PedXingState2d,
                        a::Tuple{Int64,Int64}, sp::PedXingState2d, rng::AbstractRNG, frame_j::Nullable = Nullable())
    sp_i = sp.agent_states[1]
    sp_j = sp.agent_states[2]
    pedacts = PedActionSpace_PedXing().actions

    dist_i = sp_i.state[1]
    vel_i = sp_i.state[4]
    dist_j = sp_j.y
    vel_j = pedacts[a[2]].v

    oi_dist = 1
    if dist_i > -5.0 && dist_i <= 0.0
        oi_dist = 2
    elseif dist_i > -10.0 && dist_i <= -5.0
        oi_dist = 3
    elseif dist_i > -20.0 && dist_i <= -10.0
        oi_dist = 4
    elseif dist_i <= -20.0
        oi_dist = 5
    end

    oi_vel = 1
    if vel_i > 1.0 && vel_i <= 5.0
        oi_vel = 2
    elseif vel_i > 5.0 && vel_i <= 10.0
        oi_vel = 3
    elseif vel_i > 10.0
        oi_vel = 4
    end
    o_i = (oi_dist, oi_vel)

    oj_dist = 1
    if dist_j > -2.5 && dist_j <= -2.0
        oj_dist = 2
    elseif dist_j > -2.0 && dist_j <= -1.5
        oj_dist = 3
    elseif dist_j > -1.5 && dist_j <= -1.0
        oj_dist = 4
    elseif dist_j > -1.0 && dist_j <= -0.5
        oj_dist = 5
    elseif dist_j > -0.5 && dist_j <= 0.0
        oj_dist = 6
    elseif dist_j > 0.0 && dist_j <= 0.5
        oj_dist = 7
    elseif dist_j > 0.5 && dist_j <= 1.0
        oj_dist = 8
    elseif dist_j > 1.0 && dist_j <= 1.5
        oj_dist = 9
    elseif dist_j > 1.5 && dist_j <= 2.0
        oj_dist = 10
    elseif dist_j > 2.0 && dist_j <= 2.5
        oj_dist = 11
    elseif dist_j > 2.5
        oj_dist = 12
    end

    oj_vel = 1
    if vel_j > 0.5 && vel_j <= 0.9
        oj_vel = 2
    elseif vel_j > 0.9 && vel_j <= 1.3
        oj_vel = 3
    elseif vel_j > 1.3 && vel_j <= 1.7
        oj_vel = 4
    elseif vel_j > 1.7
        oj_vel = 5
    end
    o_j = (oj_dist, oj_vel)
    return (o_i,o_j)
end

function reward(p::PedXingPOMDP_Ped, s::PedXingState2d, a::Tuple{Int64,Int64},
                    sp::PedXingState2d, rng::AbstractRNG, frame_j::Nullable = Nullable())
    if s.terminal > 0
        return 0.0
    end

    s_i = s.agent_states[1]
    s_j = s.agent_states[2]
    sp_i = sp.agent_states[1]
    sp_j = sp.agent_states[2]
    if check_collision(sp_i, sp_j)
        return p.collision_cost
    end
    caracts = VehicleActionSpace_PedXing().actions
    pedacts = PedActionSpace_PedXing().actions
    a_i = caracts[a[1]]
    a_j = pedacts[a[2]]

    reward = 0.0
    #Agent-wise reward

    if sp_j.y >= CROSSWALK_WIDTH/2.0 + 0.5 && s_j.y < CROSSWALK_WIDTH/2.0 + 0.5
        reward += p.success_reward
    end
    desired_ped_velocity = p.desired_velocity
    current_ped_velocity = a_j.v
    vel_deviation = abs(desired_ped_velocity - current_ped_velocity)

    reward -= (vel_deviation * p.vel_dev_cost_coeff)^p.vel_dev_cost_pow * p.decision_timestep

    return reward
end


function reward(p::PedXingPOMDP_Veh, s::PedXingState2d, a::Tuple{Int64,Int64},
                    sp::PedXingState2d, rng::AbstractRNG, frame_j::Nullable = Nullable())
    if s.terminal > 0
        return 0.0
    end

    s_i = s.agent_states[1]
    s_j = s.agent_states[2]
    sp_i = sp.agent_states[1]
    sp_j = sp.agent_states[2]
    if check_collision(sp_i, sp_j)
        return p.collision_cost
    end
    caracts = VehicleActionSpace_PedXing().actions
    pedacts = PedActionSpace_PedXing().actions
    a_i = caracts[a[1]]
    a_j = pedacts[a[2]]

    reward = 0.0
    if s_i.state[1] < 5.0 && sp_i.state[1] >= 5.0
        reward += p.success_reward
    end
    #Agent-wise reward
    car_vel = sp_i.state[4]
    reward -= abs((car_vel-p.desired_velocity) * p.veh_vel_dev_cost_coeff)

    ped_vel = a_j.v
    ped_frame = get(frame_j, 0)
    if typeof(ped_frame) <: AbstractIPOMDP && sp_j.y < CROSSWALK_WIDTH/2.0 && sp_j.y > -CROSSWALK_WIDTH/2.0 && sp_i.state[1] < 0.0
        desired_ped_vel = ped_frame.thisPOMDP.desired_velocity
        reward -= abs((ped_vel - desired_ped_vel) * p.ped_vel_dev_cost_coeff)^p.ped_vel_dev_cost_pow
    end
    if a_i.accl <= -6.0
        reward -= abs(p.hard_brake_cost)
    end
    return reward
end

function generate_sor(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, s::PedXingState2d,
                        a::Tuple{Int64,Int64}, rng::AbstractRNG, frame_j::Nullable = Nullable())

    sp = generate_s(p,s,a,rng, frame_j)
    o = generate_o(p,s,a,sp,rng, frame_j)
    r = reward(p,s,a,sp,rng, frame_j)
    return sp,o,r
end

function generate_sr(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, s::PedXingState2d,
                        a::Tuple{Int64,Int64}, rng::AbstractRNG, frame_j::Nullable = Nullable())

    sp = generate_s(p,s,a,rng, frame_j)
    r = reward(p,s,a,sp,rng, frame_j)
    return sp,r
end

function generate_so(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, s::PedXingState2d,
                        a::Tuple{Int64,Int64}, rng::AbstractRNG, frame_j::Nullable = Nullable())

    sp = generate_s(p,s,a,rng, frame_j)
    o = generate_o(p,s,a,sp,rng, frame_j)
    return sp,o
end

function generate_or(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, s::PedXingState2d,
                        a::Tuple{Int64,Int64}, sp::PedXingState2d, rng::AbstractRNG, frame_j::Nullable = Nullable())

    o = generate_o(p,s,a,sp,rng, frame_j)
    r = reward(p,s,a,sp,rng, frame_j)
    return o,r
end

function obs_weight(p::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, s::PedXingState2d, a::Tuple{Int64,Int64},
                    sp::PedXingState2d, o::Tuple{Tuple{Int64, Int64},Tuple{Int64, Int64}},
                    rng::AbstractRNG, frame_j::Nullable = Nullable())
    o_generated = generate_o(p,s,a,sp,rng,frame_j)
    if o == o_generated
        return 1.0
    end
    return 0.0
end

function initialize_subintentional_frame_sets(::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped})
    agent_SF_sets = Vector{Vector{Subintentional_Frame}}(2)

    veh_sf_sets = Vector{Subintentional_Frame}(2)
    ped_sf_sets = Vector{Subintentional_Frame}(1)

    #Vehicle (pesudo)-constant velocity frame
    veh_const_vel_frame = Static_Distribution_Frame(1, Dict(1 => 0.0, 2 => 0.0, 3 => 0.05, 4 => 0.9,
                                                        5 => 0.05, 6 => 0.0, 7 => 0.0, 8 => 0.0))
    num_veh_acts = length(VehicleActionSpace_PedXing().actions)
    #Vehicle (pseudo)-constant acceleration frame
    veh_const_accln_model = FSM_Simple_Frame{Int64}(1, num_veh_acts)   #One node for each acceleration
    default_tran_prob = Vector{Tuple{Int64,Float64}}(num_veh_acts)
    for i in 1:num_veh_acts
        default_tran_prob[i] = (i,0.2/num_veh_acts)
    end
    for i in 1:num_veh_acts
        tran_prob_i = copy(default_tran_prob)
        tran_prob_i[i] = (tran_prob_i[i][1], tran_prob_i[i][2]+0.8)
        node_i =  FSMNode_Simple(Dict{Int64,Float64}(i=>1.0), Dict{Int64,Vector{Tuple{Int64,Float64}}}(i=>tran_prob_i))
        veh_const_accln_model.node_vec[i] = node_i
    end
    veh_sf_sets[1] = veh_const_vel_frame
    veh_sf_sets[2] = veh_const_accln_model

    num_ped_acts = length(PedActionSpace_PedXing().actions)
    ped_const_vel_model = FSM_Simple_Frame{Int64}(2, num_ped_acts)
    default_tran_prob = Vector{Tuple{Int64,Float64}}(num_ped_acts)
    for i in 1:num_ped_acts
        default_tran_prob[i] = (i,0.2/num_ped_acts)
    end
    for i in 1:num_ped_acts
        tran_prob_i = copy(default_tran_prob)
        tran_prob_i[i] = (tran_prob_i[i][1], tran_prob_i[i][2]+0.8)
        node_i =  FSMNode_Simple(Dict{Int64,Float64}(i=>1.0), Dict{Int64,Vector{Tuple{Int64,Float64}}}(i=>tran_prob_i))
        ped_const_vel_model.node_vec[i] = node_i
    end
    ped_sf_sets[1] = ped_const_vel_model
    agent_SF_sets[1] = veh_sf_sets
    agent_SF_sets[2] = ped_sf_sets
    return agent_SF_sets
end

function initialize_intentional_frame_sets(p1::Vector{PedXingPOMDP_Veh}, p2::Vector{PedXingPOMDP_Ped})
    frame_sets = Vector{Vector{POMDP}}(2)
    frame_sets[1] = sizehint!(Vector{POMDP}(),length(p1))
    for v_p in p1
        push!(frame_sets[1],v_p)
    end
    frame_sets[2] = sizehint!(Vector{POMDP}(),length(p2))
    for p_p in p2
        push!(frame_sets[2],p_p)
    end
    return frame_sets
end

type PedXing_Frame_Distribution
    ipomdp::IPOMDP_2
    phy_state_distribution::PedXingDistribution
    subintentional_cp::Vector{Float64}
    intentional_cp::Vector{Tuple{Float64, PedXing_Frame_Distribution}}
end

function PedXing_Frame_Distribution(ipomdp::IPOMDP_2, td::PedXingDistribution)
    n_frames = length(ipomdp.oaSM) + length(ipomdp.oaFrames)    #Works for L0 as well
    sub_prob = sizehint!(Vector{Float64}(), length(ipomdp.oaSM))
    for i in 1:length(ipomdp.oaSM)
        if level(ipomdp) == 0
            push!(sub_prob, 1.0/n_frames)
        else
            push!(sub_prob, 0.0)
        end
    end
    int_prob = sizehint!(Vector{Tuple{Float64, PedXing_Frame_Distribution}}(), length(ipomdp.oaFrames))

    if level(ipomdp) > 0
        for i in 1:length(ipomdp.oaFrames)
            #push!(int_prob, (1.0/n_frames, initial_state_distribution(ipomdp.oaFrames[i])))
            push!(int_prob, (1.0/length(ipomdp.oaFrames), initial_state_distribution(ipomdp.oaFrames[i])))
        end
    end
    return PedXing_Frame_Distribution(ipomdp, td, sub_prob, int_prob)
end

#TODO: May not need to pass the pomdp if the distribution type is self identifying
function initial_ipomdp_frame_distribution(pomdp::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped},
                                            ipomdp::IPOMDP_2, pedxing_dist::PedXingDistribution)
    if !(typeof(ipomdp.thisPOMDP) <: Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped})
        println("Not an pedxing pomdp")
        return Vector{Float64}()
    end
    return PedXing_Frame_Distribution(ipomdp, pedxing_dist)
end

function num_nested_particles(pomdp::Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped}, ipomdp::IPOMDP_2)
    num_particles = sizehint!(Vector{Int64}(),level(ipomdp)+1)
    if !(typeof(ipomdp.thisPOMDP) <: Union{PedXingPOMDP_Veh, PedXingPOMDP_Ped})
        println("Not an pedxing pomdp")
        return num_particles
    end
    level(ipomdp) == 1 ? num_particles = [100,500] : num_particles = [30,100,500]
    return num_particles
end

function rand(rng::AbstractRNG, dist::PedXing_Frame_Distribution, n_particles::Vector{Int64})
    ipomdp = dist.ipomdp
    lvl = level(ipomdp)
    n_particle = n_particles[lvl+1]
    particles = sizehint!(Vector{InteractiveState{PedXingState2d}}(), n_particle)
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
        push!(particles, InteractiveState{PedXingState2d}(s,model))
    end

    return  InteractiveParticleCollection(particles)
end
