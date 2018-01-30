type Intentional_Model <: Model
    frame::AbstractIPOMDP
    belief::AbstractParticleInteractiveBelief
end

function print(model::Intentional_Model, numTabs::Int64=0)
    for i in 1:numTabs
        print("\t")
    end
    println(model.frame)
    for is in particles(model.belief)
        print(is,numTabs+1)
    end
end

function update_model{S,A,OA}(m_j::Intentional_Model, s::S, a::A, oa::OA, sp::S,
                                rng::AbstractRNG, solver::AbstractIPOMDPSolver)
    frame = m_j.frame
    belief = m_j.belief
    oj_set = observations(frame)    #TODO Unnecessary. why not generate_o followed by obs_weight
    n = n_particles(belief)

    j_ID = agentID(frame)
    i_ID = 1
    if j_ID == 1
        i_ID = 2
    end

    i_ID == 1 ? jnt_act = (a,oa) : jnt_act = (oa,a)

    ipf = SimpleInteractiveParticleFilter(frame, LowVarianceResampler(n), rng, solver)

    updated_model_probs = Dict{Model, Float64}()

    for oj in oj_set
        #order of actions reversed because frame is ipomdp for the other agent,
        p_oj = obs_weight(frame, s, oa, a, sp, oj, rng) #NOTE: first sp is a dummy
        m_jp = Intentional_Model(frame, update(ipf, belief, oa, oj))
        if haskey(updated_model_probs, m_jp)
            updated_model_probs[m_jp] += p_oj
        else
            updated_model_probs[m_jp] = p_oj
        end
    end

    return updated_model_probs
end
