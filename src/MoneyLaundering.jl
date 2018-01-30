#States: Red Team
const RS_Dirty_Pot = 1
const RS_Bank_Accts = 2
const RS_Insurance_Accts = 3
const RS_Securities = 4
const RS_Offshore_Accts = 5
const RS_Shell_Cos = 6
const RS_Trusts = 7
const RS_Corp_Loans = 8
const RS_Casino_Accts = 9
const RS_Real_Estate = 10
const RS_Clean_Pot = 11
const RS_Captured = 12

#States: Blue Team
const BS_No_Sensor = 1
const BS_Bank_Accts = 2
const BS_Insurance_Accts = 3
const BS_Securities = 4
const BS_Shell_Cos = 5
const BS_Trusts = 6
const BS_Corp_Loans = 7
const BS_Casino_Accts = 8
const BS_Real_Estate = 9


#Actions: Red Team
const RA_Placement = 1
const RA_Layering = 2
const RA_Integration = 3
const RA_Listening = 4

#Actions: Blue Team
const BA_Confiscate = 1
const BA_Bank_Accts = 2
const BA_Insurance_Accts = 3
const BA_Securities = 4
const BA_Shell_Cos = 5
const BA_Trusts = 6
const BA_Corp_Loans = 7
const BA_Casino_Accts = 8
const BA_Real_Estate = 9

#Observations: Red Team
const RO_No_Observation = 1
const RO_Bank_Accts = 2
const RO_Insurance_Accts = 3
const RO_Securities = 4
const RO_Shell_Cos = 5
const RO_Trusts = 6
const RO_Corp_Loans = 7
const RO_Casino_Accts = 8
const RO_Real_Estate = 9

#Observations: Blue Team
const BO_No_Observation = 1
const BO_Bank_Accts = 2
const BO_Insurance_Accts = 3
const BO_Securities = 4
const BO_Trusts = 5
const BO_Corp_Loans = 6
const BO_Casino_Accts = 7
const BO_Real_Estate = 8
const BO_Sensor_Alert = 9

type ML_Red_Team_Problem <: POMDP{Tuple{Int64,Int64}, Tuple{Int64,Int64}, Int64}
    r_clean_pot::Float64 # = 100
    r_capture::Float64 # = -100
    r_listen::Float64 # = -20
    r_other::Float64 # = -10
    t_sensor_placement::Float64
    t_dirty_placement_placement::Float64
    t_placement_layering_placement::Float64
    t_placement_layering_layering::Float64
    t_layering_layering_layering::Float64
    t_layering_layering_integration::Float64
    t_integration_integration_clean::Float64
    o_correct_listen::Float64
    discount_factor::Float64 # discount

    agID::Int64
end

ML_Red_Team_Problem() = ML_Red_Team_Problem(100, -100, -20, -10, 1.0, 1.0/3, 2.0/9, 1.0/9, 2.0/9, 0.5, 0.7, 0.95, 1)

type ML_Blue_Team_Problem <: POMDP{Tuple{Int64,Int64}, Tuple{Int64,Int64}, Int64}
    r_clean_pot::Float64 # = -100
    r_capture::Float64 # = 100
    r_false_confiscate::Float64 # = -25
    r_other::Float64 # = -10
    t_sensor_placement::Float64
    t_placement_layering_placement::Float64
    t_placement_layering_layering::Float64
    t_layering_layering_layering::Float64
    t_layering_layering_integration::Float64
    t_integration_integration_clean::Float64
    o_correct_report::Float64
    o_correct_sensor_reading::Float64
    discount_factor::Float64 # discount

    agID::Int64
end
ML_Blue_Team_Problem() = ML_Blue_Team_Problem(-100, 100, -25, -10, 1.0, 1.0/3, 2.0/9, 1.0/9, 2.0/9, 1.0/9, 0.5, 0.7, 1.0, 0.95, 2)


discount(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}) = p.discount_factor

function isterminal(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}, st::Tuple{Int64,Int64})
  return st[1] == RS_Clean_Pot || st[1] == RS_Captured
end

n_states(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}) = 108
n_actions(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}) = length(actions(p))
actions(::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}) = [(RA_Placement,BA_Confiscate),    (RA_Placement,BA_Place_Bank_Accts),   (RA_Placement,BA_Place_Insurance_Accts),   (RA_Placement,BA_Place_Securities),   (RA_Placement,BA_Place_Shell_Cos),   (RA_Placement,BA_Place_Trusts),   (RA_Placement,BO_Corp_Loans),   (RA_Placement,RO_Casino_Accts),   (RA_Placement,BA_Place_Real_Estate),
                                                                (RA_Layering,BA_Confiscate),    (RA_Layering,BA_Place_Bank_Accts),    (RA_Layering,BA_Place_Insurance_Accts),    (RA_Layering,BA_Place_Securities),    (RA_Layering,BA_Place_Shell_Cos),    (RA_Layering,BA_Place_Trusts),    (RA_Layering,BO_Corp_Loans),    (RA_Layering,RO_Casino_Accts),    (RA_Layering,BA_Place_Real_Estate),
                                                                (RA_Integration,BA_Confiscate), (RA_Integration,BA_Place_Bank_Accts), (RA_Integration,BA_Place_Insurance_Accts), (RA_Integration,BA_Place_Securities), (RA_Integration,BA_Place_Shell_Cos), (RA_Integration,BA_Place_Trusts), (RA_Integration,BO_Corp_Loans), (RA_Integration,RO_Casino_Accts), (RA_Integration,BA_Place_Real_Estate),
                                                                (RA_Listening,BA_Confiscate),   (RA_Listening,BA_Place_Bank_Accts),   (RA_Listening,BA_Place_Insurance_Accts),   (RA_Listening,BA_Place_Securities),   (RA_Listening,BA_Place_Shell_Cos),   (RA_Listening,BA_Place_Trusts),   (RA_Listening,BO_Corp_Loans),   (RA_Listening,RO_Casino_Accts),   (RA_Listening,BA_Place_Real_Estate)]
action_index(::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}, a::Tuple{Int64, Int64}) = (a[1]-1)*9 + a[2]
observations(::ML_Red_Team_Problem)  = [RO_No_Observation,RO_Bank_Accts,RO_Insurance_Accts,RO_Securities,RO_Shell_Cos,RO_Trusts,RO_Corp_Loans,RO_Casino_Accts,RO_Real_Estate]
observations(::ML_Blue_Team_Problem) = [BO_No_Observation,BO_Bank_Accts,BO_Insurance_Accts,BO_Securities,BO_Trusts,BO_Corp_Loans,BO_Casino_Accts,BO_Real_Estate,BO_Sensor_Alert]

type MLDistribution
    p::Float64
    #it::Vector{Int64}
end
MLDistribution() = MLDistribution(1.0)

function initial_state_distribution(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem})
    return MLDistribution()
end
function rand(rng, dist::MLDistribution)
    s = 0
    rand(rng) < dist.p ? s=1 : s = rand(2:108)
    return s
end

function generate_s(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    #generate s
    sp = s

    a_red = a[1]
    a_blue = a[2]

    s_red = s[1]
    s_blue = s[2]

    sp_red = s_red
    sp_blue = s_blue
    if a_blue == BA_Confiscate
        if (s_red == RS_Bank_Accts && s_blue == BS_Bank_Accts)
            || (s_red == RS_Insurance_Accts && s_blue == BS_Insurance_Accts)
            || (s_red == RS_Securities && s_blue == BS_Securities)
            || (s_red == RS_Securities && s_blue == BS_Securities)
            || (s_red == RS_Shell_Cos && s_blue == BS_Shell_Cos)
            || (s_red == RS_Trusts && s_blue == BS_Trusts)
            || (s_red == RS_Corp_Loans && s_blue == BS_Corp_Loans)
            || (s_red == RS_Casino_Accts && s_blue == BS_Casino_Accts)
            || (s_red == RS_Real_Estate && s_blue == BS_Real_Estate)

            sp_red = RS_Captured
            return (sp_red, sp_blue)
        end
    end
    if a_blue == BA_Bank_Accts
        sp_blue = BS_Bank_Accts
    elseif a_blue == BA_Insurance_Accts
        sp_blue = BS_Insurance_Accts
    elseif a_blue == BA_Securities
        sp_blue = BS_Securities
    elseif a_blue == BA_Shell_Cos
        sp_blue = BS_Shell_Cos
    elseif a_blue == BA_Trusts
        sp_blue = BS_Trusts
    elseif a_blue == BA_Corp_Loans
        sp_blue = BS_Corp_Loans
    elseif a_blue == BA_Casino_Accts
        sp_blue = BS_Casino_Accts
    elseif a_blue == BA_Real_Estate
        sp_blue = BS_Real_Estate
    end

    if a_red == RA_Placement
        if s_red == RS_Dirty_Pot
            rnd = rand(rng)
            if rnd < p.t_dirty_placement_placement
                sp_red = RS_Bank_Accts
            elseif rnd < 2 * p.t_dirty_placement_placement
                sp_red = RS_Insurance_Accts
            else
                sp_red = RS_Securities
            end
        end
    elseif a_red == RA_Layering
        # If placement states
        if s_red == RS_Bank_Accts || s_red == RS_Insurance_Accts || s_red == RS_Securities
            rnd = rand(rng)
            if rnd <= p.t_placement_layering_placement
                sp_red = RS_Bank_Accts
            else
                rnd -= p.t_placement_layering_placement
                if rnd <= p.t_placement_layering_placement
                    sp_red = RS_Insurance_Accts
                else
                    rnd -= p.t_placement_layering_placement
                    if rnd <= p.t_placement_layering_placement
                        sp_red = RS_Securities
                    else
                        rnd -= p.t_placement_layering_placement
                        if rnd <= p.t_placement_layering_layering
                            sp_red = RS_Offshore_Accts
                        else
                            rnd -= p.t_placement_layering_layering
                            if rnd <= p.t_placement_layering_layering
                                sp_red = RS_Shell_Cos
                            else
                                sp_red = RS_Trusts
                            end
                        end
                    end
                end
            end

        elseif s_red == RS_Offshore_Accts || s_red == RS_Shell_Cos || s_red == RS_Trusts
            rnd = rand(rng)
            if rnd <= p.t_layering_layering_layering
                sp_red = RS_Offshore_Accts
            else
                rnd -= p.t_layering_layering_layering
                if rnd <= p.t_layering_layering_layering
                    sp_red = RS_Shell_Cos
                else
                    rnd -= p.t_layering_layering_layering
                    if rnd <= p.t_layering_layering_layering
                        sp_red = RS_Trusts
                    else
                        rnd -= p.t_layering_layering_layering
                        if rnd <= p.t_layering_layering_integration
                            sp_red = RS_Corp_Loans
                        else
                            rnd -= p.t_layering_layering_integration
                            if rnd <= p.t_layering_layering_integration
                                sp_red = RS_Casino_Accts
                            else
                                sp_red = RS_Real_Estate
                            end
                        end
                    end
                end
            end
        end
    elseif a_red == RA_Integration
        if s_red == RS_Corp_Loans || s_red == RS_Casino_Accts || s_red == RS_Real_Estate
            rnd = rand(rng)
            if rnd <= p.t_integration_integration_clean
                sp_red = RS_Clean_Pot
            end
        end
    elseif a_red == RA_Listening

    end
    return (sp_red,sp_blue)
end

function generate_o(p::ML_Red_Team_Problem, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, sp::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    a_red = a[1]
    a_blue = a[2]

    s_red = sp[1]
    s_blue = sp[2]

    if a_red != RA_Listening
        return RO_No_Observation
    end
    rnd = rand(rng)
    if rnd <= p.o_correct_listen
        if s_blue == BS_Bank_Accts
            return RO_Bank_Accts
        elseif s_blue == BS_Insurance_Accts
            return RO_Insurance_Accts
        elseif s_blue == BS_Securities
            return RO_Securities
        elseif s_blue == BS_Shell_Cos
            return RO_Shell_Cos
        elseif s_blue == BS_Trusts
            return RO_Trusts
        elseif s_blue == BS_Corp_Loans
            return RO_Corp_Loans
        elseif s_blue == BS_Casino_Accts
            return RO_Casino_Accts
        elseif s_blue == BS_Real_Estate
            return RO_Real_Estate
        end
    else
        return RO_No_Observation
    end
    return RO_No_Observation
end

function generate_o(p::ML_Blue_Team_Problem, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, sp::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    a_red = a[1]
    a_blue = a[2]

    s_red = sp[1]
    s_blue = sp[2]
    if (a_blue == BA_Bank_Accts && s_red == RS_Bank_Accts)
        || (a_blue == BA_Insurance_Accts && s_red == RS_Insurance_Accts)
        || (a_blue == BA_Securities && s_red == RS_Securities)
        || (a_blue == BA_Shell_Cos && s_red == RS_Shell_Cos)
        || (a_blue == BA_Trusts && s_red == RS_Trusts)
        || (a_blue == BA_Corp_Loans && s_red == RS_Corp_Loans)
        || (a_blue == BA_Casino_Accts && s_red == RS_Casino_Accts)
        || (a_blue == BA_Real_Estate && s_red == RS_Real_Estate)

        return BO_Sensor_Alert
    end
    if rand(rng) < p.o_correct_listen
        if s_red == RS_Bank_Accts
            return BO_Bank_Accts
        elseif s_red == RS_Insurance_Accts
            return BO_Insurance_Accts
        elseif s_red == RS_Securities
            return BO_Securities
        elseif s_red == RS_Trusts
            return BO_Trusts
        elseif s_red == RS_Corp_Loans
            return BO_Corp_Loans
        elseif s_red == RS_Casino_Accts
            return BO_Casino_Accts
        elseif s_red == RS_Real_Estate
            return BO_Real_Estate
        end
    end
    return BO_No_Observation
end

function reward(p::ML_Red_Team_Problem, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    #generate s
    sp = s

    a_red = a[1]
    a_blue = a[2]

    s_red = s[1]
    s_blue = s[2]

    if a_blue == BA_Confiscate
        if (s_red == RS_Bank_Accts && s_blue == BS_Bank_Accts)
            || (s_red == RS_Insurance_Accts && s_blue == BS_Insurance_Accts)
            || (s_red == RS_Securities && s_blue == BS_Securities)
            || (s_red == RS_Securities && s_blue == BS_Securities)
            || (s_red == RS_Shell_Cos && s_blue == BS_Shell_Cos)
            || (s_red == RS_Trusts && s_blue == BS_Trusts)
            || (s_red == RS_Corp_Loans && s_blue == BS_Corp_Loans)
            || (s_red == RS_Casino_Accts && s_blue == BS_Casino_Accts)
            || (s_red == RS_Real_Estate && s_blue == BS_Real_Estate)

            return p.r_capture
        end
    end

    if s_red == RS_Clean_Pot
        return p.r_clean_pot
    end

    if a_red == RA_Listening
        return p.r_listen
    end
    return p.r_other
end

function reward(p::ML_Blue_Team_Problem, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    #generate s
    sp = s

    a_red = a[1]
    a_blue = a[2]

    s_red = s[1]
    s_blue = s[2]

    if a_blue == BA_Confiscate
        if (s_red == RS_Bank_Accts && s_blue == BS_Bank_Accts)
            || (s_red == RS_Insurance_Accts && s_blue == BS_Insurance_Accts)
            || (s_red == RS_Securities && s_blue == BS_Securities)
            || (s_red == RS_Securities && s_blue == BS_Securities)
            || (s_red == RS_Shell_Cos && s_blue == BS_Shell_Cos)
            || (s_red == RS_Trusts && s_blue == BS_Trusts)
            || (s_red == RS_Corp_Loans && s_blue == BS_Corp_Loans)
            || (s_red == RS_Casino_Accts && s_blue == BS_Casino_Accts)
            || (s_red == RS_Real_Estate && s_blue == BS_Real_Estate)

            return p.r_capture
        else
            return p.r_false_confiscate
        end
    end

    if s_red == RS_Clean_Pot
        return p.r_clean_pot
    end

    return p.r_other
end
function generate_or(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, sp::Tuple{Int64,Int64}, rng::AbstractRNG)
    o = generate_o(p,s,a,sp,rng)
    r = reward(p,s,a,rng)

    return o,r
end

function obs_weight(p::ML_Red_Team_Problem, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, sp::Tuple{Int64,Int64}, o::Int64, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    a_red = a[1]
    a_blue = a[2]

    s_red = sp[1]
    s_blue = sp[2]

    if a_red != RA_Listening
        if o == RO_No_Observation
            return 1.0
        else
            return 0.0
        end
    end

    if (s_blue == BS_Bank_Accts && o == RO_Bank_Accts)
        || (s_blue == BS_Insurance_Accts && o == RO_Insurance_Accts)
        || (s_blue == BS_Securities && o == RO_Securities)
        || (s_blue == BS_Shell_Cos && o == RO_Shell_Cos)
        || (s_blue == BS_Trusts && o == RO_Trusts)
        || (s_blue == BS_Corp_Loans && o == RO_Corp_Loans)
        || (s_blue == BS_Casino_Accts && o == RO_Casino_Accts)
        || (s_blue == BS_Real_Estate && o == RO_Real_Estate)

        return p.o_correct_listen
    elseif o == RO_No_Observation
        return 1.0 - p.o_correct_listen
    else
        return 0.0
    end

end

function obs_weight(p::ML_Blue_Team_Problem, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, sp::Tuple{Int64,Int64}, o::Int64, rng::AbstractRNG)
    agID = p.agID
    oaID=1
    if agID == 1
        oaID = 2
    end

    a_red = a[1]
    a_blue = a[2]

    s_red = sp[1]
    s_blue = sp[2]

    if (a_blue == BA_Bank_Accts && s_red == RS_Bank_Accts)
        || (a_blue == BA_Insurance_Accts && s_red == RS_Insurance_Accts)
        || (a_blue == BA_Securities && s_red == RS_Securities)
        || (a_blue == BA_Shell_Cos && s_red == RS_Shell_Cos)
        || (a_blue == BA_Trusts && s_red == RS_Trusts)
        || (a_blue == BA_Corp_Loans && s_red == RS_Corp_Loans)
        || (a_blue == BA_Casino_Accts && s_red == RS_Casino_Accts)
        || (a_blue == BA_Real_Estate && s_red == RS_Real_Estate)

        if o == BO_Sensor_Alert
            return 1.0
        else
            return 0.0
        end
    end

    if (s_red == RS_Bank_Accts && o = BO_Bank_Accts)
        || (s_red == RS_Insurance_Accts && o = BO_Insurance_Accts)
        || (s_red == RS_Securities && o = BO_Securities)
        || (s_red == RS_Trusts && o = BO_Trusts)
        || (s_red == RS_Corp_Loans && o = BO_Corp_Loans)
        || (s_red == RS_Casino_Accts && o = BO_Casino_Accts)
        || (s_red == RS_Real_Estate && o = BO_Real_Estate)

        return p.o_correct_listen
    elseif o == BO_No_Observation
        return 1.0 - p.o_correct_listen
    end
    return 0.0
end

function generate_sor(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    o = generate_o(p, s, a, sp, rng)
    r = reward(p,s,a,rng)
    return sp,o,r
end

function generate_so(p::Union{ML_Red_Team_Problem, ML_Blue_Team_Problem}, s::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    o = generate_o(p, s, a, sp, rng)
    return sp,o
end
