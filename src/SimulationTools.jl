function simulate(sim::HistoryRecorder, ipomdp_i::IPOMDP_2, policy_i::Policy,
                                        ipomdp_j::IPOMDP_2, policy_j::Policy)
    up_i = updater(policy_i)
    up_j = updater(policy_j)
    simulate(sim, ipomdp_i, policy_i, up_i, ipomdp_j, policy_j, up_j)
end

function simulate(sim::HistoryRecorder, ipomdp_i::IPOMDP_2, policy_i::Policy, bu_i::Updater, ipomdp_j::IPOMDP_2, policy_j::Policy, bu_j::Updater)
    dist_i = initial_state_distribution(ipomdp_i)
    dist_j = initial_state_distribution(ipomdp_j)
    simulate(sim, ipomdp_i, policy_i, bu_i, dist_i, ipomdp_j, policy_j, bu_j, dist_j)
end

function simulate(sim::HistoryRecorder, ipomdp_i::IPOMDP_2, policy_i::Policy, bu_i::Updater, dist_i::Any,
                        ipomdp_j::IPOMDP_2, policy_j::Policy, bu_j::Updater, dist_j::Any)
    rng = sim.rng
    initial_state = rand(rng, initial_state_distribution(ipomdp_i.thisPOMDP))

    #TODO: Need better names for other kinds of belief, may be belief parameters
    num_particles_i = num_nested_particles(ipomdp_i.thisPOMDP, ipomdp_i)
    num_particles_j = num_nested_particles(ipomdp_j.thisPOMDP, ipomdp_j)

    initial_belief_i = rand(rng, dist_i, num_particles_i)
    initial_belief_j = rand(rng, dist_j, num_particles_j)

    max_steps = get(sim.max_steps, typemax(Int))
    if !isnull(sim.eps)
        max_steps = min(max_steps, ceil(Int,log(get(sim.eps))/log(discount(pomdp))))
    end
    sizehint = get(sim.sizehint, min(max_steps, 1000))

    S = state_type(ipomdp_i)
    A_i = action_type(ipomdp_i)
    A_j = action_type(ipomdp_j)
    O_i = obs_type(ipomdp_i)
    O_j = obs_type(ipomdp_j)

    # aliases for the histories to make the code more concise
    sh = sizehint!(Vector{S}(0), sizehint)
    a_ih = sizehint!(Vector{A_i}(0), sizehint)
    a_jh = sizehint!(Vector{A_j}(0), sizehint)
    o_ih = sizehint!(Vector{O_i}(0), sizehint)
    o_jh = sizehint!(Vector{O_j}(0), sizehint)
    b_ih = sizehint!(Vector{typeof(initial_belief_i)}(0), sizehint)
    b_jh = sizehint!(Vector{typeof(initial_belief_j)}(0), sizehint)
    r_ih = sizehint!(Vector{Float64}(0), sizehint)
    r_jh = sizehint!(Vector{Float64}(0), sizehint)
    #TODO: Use these fields later. Ignoring for now
    infoh = sizehint!(Vector{Any}(0), sizehint)
    ainfoh = sizehint!(Vector{Any}(0), sizehint)
    uinfoh = sizehint!(Vector{Any}(0), sizehint)
    exception = Nullable{Exception}()
    backtrace = Nullable{Any}()

    push!(sh, initial_state)
    push!(b_ih, initial_belief_i)
    push!(b_jh, initial_belief_j)

    if sim.show_progress
        if isnull(sim.max_steps) && isnull(sim.eps)
            error("If show_progress=true in a HistoryRecorder, you must also specify max_steps or eps.")
        end
        prog = Progress(max_steps, "Simulating..." )
    end

    disc = 1.0
    step = 1
    i_planning_time = 0.0
    try
        while step <= max_steps
            if isterminal(ipomdp_i.thisPOMDP, sh[step])
                break
            end
            #println("Step: ",step)
            t1 = time_ns()
            a_i = action(policy_i, b_ih[step])
            t2 = time_ns()
            i_planning_time += (t2 - t1)/1.0e9
            #println("Actions: ",a_i)
            a_j = action(policy_j, b_jh[step])
            #println("Actions: ",a_i," ", a_j)
            push!(a_ih, a_i)
            push!(a_jh, a_j)

            sp,o_i,r_i = generate_sor(ipomdp_i, sh[step], a_ih[step], a_jh[step], rng)
            o_j,r_j = generate_or(ipomdp_j,sh[step],a_jh[step],a_ih[step],sp,rng)

            #println("r_i = ",r_i," r_j = ",r_j)

            push!(sh, sp)
            push!(o_ih, o_i)
            push!(o_jh, o_j)
            push!(r_ih, r_i)
            push!(r_jh, r_j)

            replenish_state = Nullable{S}(sp)
            #println("Updating i from simulations")
            b_ip = update(bu_i, b_ih[step], a_ih[step], o_ih[step], replenish_state)
            #println("Updating j from simulations")
            b_jp = update(bu_j, b_jh[step], a_jh[step], o_jh[step], replenish_state)

            push!(b_ih, b_ip)
            push!(b_jh, b_jp)

            step += 1

            if sim.show_progress
                next!(prog)
            end
        end
    catch ex
        if sim.capture_exception
            exception = Nullable{Exception}(ex)
            backtrace = Nullable{Any}(catch_backtrace())
        else
            rethrow(ex)
        end
    end
    if sim.show_progress
        finish!(prog)
    end
    history_i = POMDPHistory(sh, a_ih, o_ih, b_ih, r_ih, discount(ipomdp_i), exception, backtrace)
    history_j = POMDPHistory(sh, a_jh, o_jh, b_jh, r_jh, discount(ipomdp_j), exception, backtrace)
    return history_i,history_j, i_planning_time
end

#=function reset_belief{S}(b_i::AbstractParticleInteractiveBelief, ipomdp_i::IPOMDP_2, sp::S, proportion::Float64)
    n = round(Int64, n_particles(b_i)*proportion)
    for i in 1:n

    end
end=#
