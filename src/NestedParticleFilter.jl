type InteractiveParticleCollection{T<:InteractiveState} <: AbstractParticleInteractiveBelief{T}
    particleCollection::ParticleCollection{T}
end

InteractiveParticleCollection{T<:InteractiveState}(p::AbstractVector{T}) =
                                InteractiveParticleCollection(ParticleCollection{T}(p))

n_particles(b::InteractiveParticleCollection) = n_particles(b.particleCollection)
particles(b::InteractiveParticleCollection) = particles(b.particleCollection)
weighted_particles(p::InteractiveParticleCollection) = weighted_particles(b.particleCollection)
weight_sum(::InteractiveParticleCollection) = 1.0
weight(b::InteractiveParticleCollection, i::Int) = weight(b.particleCollection,i)
particle(b::InteractiveParticleCollection, i::Int) = particle(b.particleCollection,i)
rand(rng::AbstractRNG, b::InteractiveParticleCollection) = rand(rng, b.particleCollection)
mean(b::InteractiveParticleCollection) = mean(b.particleCollection)
iterator(b::InteractiveParticleCollection) = iterator(b.particleCollection)

function get_probs{S}(b::AbstractParticleInteractiveBelief{S})
    if isnull(b.particleCollection._probs)
        # update the cache
        probs = Dict{S, Float64}()
        for (i,p) in enumerate(particles(b))
            if haskey(probs, p)
                probs[p] += weight(b, i)/weight_sum(b)
            else
                probs[p] = weight(b, i)/weight_sum(b)
            end
        end
        b.particleCollection._probs = Nullable(probs)
    end
    return get(b.particleCollection._probs)
end

pdf{S}(b::AbstractParticleInteractiveBelief{S}, s::S) = get(get_probs(b), s, 0.0)

type SimpleInteractiveParticleFilter{S<:InteractiveState, R, RNG<:AbstractRNG} <: Updater
    ipomdp::IPOMDP_2
    resample::R
    rng::RNG
    _particle_memory::Vector{S}
    _weight_memory::Vector{Float64}
    solver::AbstractIPOMDPSolver

    SimpleInteractiveParticleFilter{S, R, RNG}(model, resample, rng, solver::AbstractIPOMDPSolver) where {S,R,RNG} = new(model, resample, rng, state_type(model)[], Float64[], solver)
end
function SimpleInteractiveParticleFilter{R}(ipomdp::IPOMDP_2, resample::R, rng::AbstractRNG, solver::AbstractIPOMDPSolver)
    return SimpleInteractiveParticleFilter{InteractiveState{state_type(ipomdp)},R,typeof(rng)}(ipomdp, resample, rng, solver)
end

#Input is an interactive particle filter, set of interactive particles, and action and observation of subject agent.
function update{S}(up::SimpleInteractiveParticleFilter, bel::InteractiveParticleCollection{S}, a, o)
    ipomdp = up.ipomdp
    ipomdpsolver = up.solver
    pomdpsolver = getsolver(ipomdpsolver,level(ipomdp))
    n_aj = n_oactions(ipomdp)
    ps = particles(bel)
    pm = up._particle_memory
    wm = up._weight_memory
    rng = up.rng
    resize!(pm, 0)
    resize!(wm, 0)
    #TODO: Need a better way to get n_oj
    sizehint!(pm, n_particles(bel)*n_aj*50)
    sizehint!(wm, n_particles(bel)*n_aj*50)
    all_terminal = true

    for i in 1:n_particles(bel)
        is = ps[i]
        s = is.env_state
        m_j = is.model
        local aj::oaction_type(up.ipomdp)

        if !isterminal(up.ipomdp.thisPOMDP, s)
            all_terminal = false
            if typeof(m_j) <: Intentional_Model
                frame_j = m_j.frame
                b_j = m_j.belief
                level_j = level(frame_j)
                #solver_j = up.solver.solvers[level_j][1]
                j_planner = solve(up.solver, frame_j)
                aj_prob = actionProb(j_planner, b_j)
                #aj = action(j_planner, b_j)
            else
                frame_j = m_j.frame
                hist_j = m_j.history
                j_solver = solver(frame_j, rng=rng) #NOTE: Other Subintentional models should implement the same function calls
                j_planner = solve(j_solver, frame_j)
                aj_prob = actionProb(j_planner, hist_j)
                #aj = action(j_planner, hist_j)
            end

            for (aj,p_aj) in aj_prob
                sp,dummy_oi,r = generate_sor(up.ipomdp, s, a, aj, rng)
                pr_o = obs_weight(up.ipomdp, s, a, aj, sp, o, rng)
                updated_model_probs = update_model(m_j, s, a, aj, sp, rng, up.solver)
                #modelp = rand(updated_model_probs,rng)
                for (modelp, oj_prob) in updated_model_probs
                    isp = InteractiveState(sp,modelp)
                    push!(pm, isp)
                    push!(wm, pr_o*p_aj*oj_prob)
                end
            end
        end
    end
    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
        # TODO: create a mechanism to handle this failure
    end
    return InteractiveParticleCollection(resample(up.resample, WeightedParticleBelief{S}(pm, wm, sum(wm), nothing), up.rng))
end

function Base.srand(f::SimpleInteractiveParticleFilter, seed)
    srand(f.rng, seed)
    return f
end

function print(belief::InteractiveParticleCollection, numTabs::Int64=0)
    for is in particles(belief)
        print(is,numTabs)
        println()
    end
end

function get_physical_state_probability(belief::InteractiveParticleCollection)
    st_prob = Dict{Any,Float64}()
    n_particle = n_particles(belief)
    for is in particles(belief)
        s = is.env_state
        if !haskey(st_prob, s)
            st_prob[s] = 1.0/n_particle
        else
            st_prob[s] += 1.0/n_particle
        end
    end
    return st_prob
end
