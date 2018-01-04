mutable struct IPOMCPPlanner{P<:IPOMDP_2, SE, RNG} <: Policy
    solver::IPOMCPSolver
    problem::P
    solved_estimator::SE	#For rollout
    rng::RNG
    _best_node_mem::Vector{Int}
    _tree::Nullable
end

function IPOMCPPlanner(solver::IPOMCPSolver, ipomdp::IPOMDP_2)
    pomcpsolver = solver.solvers[ipomdp.level+1][1]
    se = BasicPOMCP.convert_estimator(pomcpsolver.estimate_value, pomcpsolver, ipomdp.thisPOMDP) #That is why I still have POMCPSolver
    return IPOMCPPlanner(solver, ipomdp, se, pomcpsolver.rng, Int[], Nullable())
end

Base.srand(p::IPOMCPPlanner, seed) = srand(p.rng, seed)

solve(solver::IPOMCPSolver, ipomdp::IPOMDP_2) = IPOMCPPlanner(solver, ipomdp)   #Is there a better way?

function updater(p::IPOMCPPlanner)
    ipomdp = p.problem
    pomcpsolver = p.solver.solvers[ipomdp.level+1][1]
    n_particles = p.solver.solvers[ipomdp.level+1][2]
    #=P = typeof(p.problem)
    S = state_type(P)
    A = action_type(P)
    O = obs_type(P)
    if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
        return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    end=#
    return SimpleInteractiveParticleFilter(ipomdp, LowVarianceResampler(n_particles), p.rng, p.solver)	# This may need to change. Because IPF needed
end

function action(p::IPOMCPPlanner, b::AbstractParticleInteractiveBelief)
    ipomdp = p.problem
    pomcpsolver = p.solver.solvers[ipomdp.level+1][1]
    local a::action_type(p.problem)
    rng = p.rng

    try
        tree = POMCPTree(ipomdp, pomcpsolver.tree_queries)

		a = search(p, b, tree)	#quantal response model for action probability

		p._tree = Nullable(tree)
	catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(action_type(p.problem), default_action(pomcpsolver.default_action, p.problem, b, ex))
    end
	return a
end

function search(p::IPOMCPPlanner, b::AbstractParticleInteractiveBelief, t::BasicPOMCP.POMCPTree)
    ipomdp = p.problem
    pomcpsolver = p.solver.solvers[ipomdp.level+1][1]
    all_terminal = true
    start_us = CPUtime_us()
    for i in 1:pomcpsolver.tree_queries							# 1 to num iter
        if CPUtime_us() - start_us >= 1e6*pomcpsolver.max_time
            break
        end
        is = rand(p.rng, b)										# returns an interactive state implemented
        if !isterminal(p.problem.thisPOMDP, is.env_state)
            simulate(p, is, POMCPObsNode(t, 1), pomcpsolver.max_depth)
            all_terminal = false
        end
    end

    if all_terminal
        throw(AllSamplesTerminal(b))
    end
    lambda = p.solver.solvers[p.problem.level+1][3]

    actValue = Dict{action_type(p.problem),Float64}()
    for node in t.children[h]
        actValue[t.a_labels[node]] = t.v[node]
    end
    actProb = quantal_response_probability(actValue, lambda)

    return rand(actProb, rng = p.rng)
end

function simulate(p::IPOMCPPlanner, is::AbstractInteractiveState, hnode::BasicPOMCP.POMCPObsNode, steps::Int)
    s = env_state(is)
    mj = model(is)
    agID = agentID(p.problem)
    oaID = 1
    if agID == 1
        oaID = 2
    end
    if steps == 0 || isterminal(p.problem.thisPOMDP, s)
        return 0.0
    end

    local aj::oaction_type(p.problem)
    if typeof(mj) <: Intentional_Model
        b_j = mj.belief
        mj_frame = mj.frame
        #mj_solver = p.solver.solvers[mj_frame.level+1][1]
        j_planner = solve(p.solver, mj_frame)
        aj = action(j_planner, b_j)
    else
        frame_j = m_j.frame
        hist_j = m_j.history
        j_solver = solver(frame_j, rng=p.rng) #NOTE: Other Subintentional models should implement the same function calls
        j_planner = solve(j_solver, frame_j)
        aj = action(j_planner, hist_j)
    end

    t = hnode.tree
    h = hnode.node													#This is the index of the current node in the tree

    ltn = log(t.total_n[h])
    best_nodes = empty!(p._best_node_mem)
    best_criterion_val = -Inf
    for node in t.children[h] 										# children of the node indexed h corresponding to each action
        n = t.n[node]												# number of times the child node has been visited
        if n == 0 && ltn <= 0.0										#Only in the first iteration
            criterion_value = t.v[node]								#Initial value
        elseif n == 0 && t.v[node] == -Inf							#Don't know why, only if it was set to -Inf I think
            criterion_value = Inf
        else
            criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    ha = rand(p.rng, best_nodes)									#random best node index
    a = t.a_labels[ha]												#action label

    agID == 1 ? jnt_act = (a,aj) : jnt_act = (aj, a)

    sp, o, r = generate_sor(p.problem, s, jnt_act, p.rng)
    mjp = rand(update_model(mj, s, a, aj, sp, p.rng), rng=p.rng)

    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        hao = insert_obs_node!(t, p.problem, ha, o)
        v = estimate_value(p.solved_estimator,
                           p.problem,
                           sp,
                           POMCPObsNode(t, hao),
                           steps-1)
        R = r + discount(p.problem)*v
    else
        isp = InteractiveState(sp, mjp)
        R = r + discount(p.problem)*simulate(p, isp, POMCPObsNode(t, hao), steps-1) * prob
    end

    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]

    return R
end
