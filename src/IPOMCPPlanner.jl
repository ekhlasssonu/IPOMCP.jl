mutable struct IPOMCPPlanner{P<:IPOMDP_2, SE, RNG} <: Policy
    solver::IPOMCPSolver
    problem::P
    solved_estimator::SE    #For rollout. ASK about it
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
    pomcpsolver = getsolver(p.solver,ipomdp.level)
    n_particle = n_particles(p.solver,ipomdp.level)
    return SimpleInteractiveParticleFilter(ipomdp, LowVarianceResampler(n_particle), p.rng, p.solver)	# This may need to change. Because IPF needed
end

function actionProb(p::IPOMCPPlanner, b::AbstractParticleInteractiveBelief)
    ipomdp = p.problem
    pomcpsolver = p.solver.solvers[ipomdp.level+1][1]
    local actProb::Dict{action_type(p.problem),Float64}
    try
        tree = POMCPTree(ipomdp, pomcpsolver.tree_queries)
        if isnull(b.act_prob)
            #println("going to search from actionProb")
            b.act_prob = Nullable(search(p,b,tree))
            p._tree = Nullable(tree)
        end

        actProb = get(b.act_prob)

    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(action_type(p.problem), default_action(pomcpsolver.default_action, p.problem, b, ex))
        actProb = Dict(a=>1.0)
    end

    return actProb
end

function action(p::IPOMCPPlanner, b::AbstractParticleInteractiveBelief)
    ipomdp = p.problem
    pomcpsolver = getsolver(p.solver,ipomdp.level)
    local a::action_type(p.problem)
    rng = p.rng
    #println("********** Getting action for agent $(agentID(ipomdp)) @ level $(level(ipomdp))**********")
    #sparse_print(b)
    #println();
    try
        tree = POMCPTree(ipomdp, pomcpsolver.tree_queries)

        if isnull(b.act_prob)
            #println("going to search")
            b.act_prob = Nullable(search(p,b,tree))
            p._tree = Nullable(tree)
        #else
            #println("Not Null actProb")
        end
        actProb = get(b.act_prob)

        a = rand(actProb, rng)
	catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(action_type(p.problem), default_action(pomcpsolver.default_action, p.problem, b, ex))
    end
    #println("**********Action for agent $(agentID(ipomdp)) @ level $(level(ipomdp)) is $a**********")
	return a
end

function search(p::IPOMCPPlanner, b::AbstractParticleInteractiveBelief, t::BasicPOMCP.POMCPTree)
    ipomdp = p.problem
    #println("\tIn Search. agent $(agentID(ipomdp)), level $(level(ipomdp))")
    pomcpsolver = p.solver.solvers[ipomdp.level+1][1]
    all_terminal = true
    start_us = CPUtime_us()
    for i in 1:pomcpsolver.tree_queries							# 1 to num iter
        if CPUtime_us() - start_us >= 1e6*pomcpsolver.max_time
            break
        end
        is = rand(p.rng, b)										# returns an interactive state implemented
        if !isterminal(p.problem.thisPOMDP, is.env_state)
            #println("\tGoing to simulate. agent $(agentID(ipomdp)), level $(level(ipomdp))")
            simulate(p, is, BasicPOMCP.POMCPObsNode(t, 1), pomcpsolver.max_depth)
            all_terminal = false
        end
    end

    if all_terminal
        return Dict(actions(ipomdp)[1]=>1.0)
        #throw(AllSamplesTerminal(b))
    end
    lambda = qr_constant(p.solver,p.problem.level)

    actValue = Dict{action_type(p.problem),Float64}()
    #println("Accessing values")
    h = 1
    for node in t.children[h]
        #print(t.a_labels[node],":",t.v[node]," ")
        actValue[t.a_labels[node]] = t.v[node]
    end
    #println(actValue)
    #println("lambda=$lambda")
    b.act_value = Nullable(actValue)
    actProb = quantal_response_probability(actValue, lambda)
    #println("Action Probs ", actProb)
    return actProb
end

function simulate(p::IPOMCPPlanner, is::AbstractInteractiveState, hnode::BasicPOMCP.POMCPObsNode, steps::Int)
    s = env_state(is)
    mj = model(is)
    agID = agentID(p.problem)
    oaID = 1
    if agID == 1
        oaID = 2
    end
    #println("\t\ts = ",s)
    if steps == 0 || isterminal(p.problem.thisPOMDP, s)
        #println("\t\tsteps = ",steps, "terminal = ",isterminal(p.problem.thisPOMDP, s))
        return 0.0
    end

    #println("\t\t In simulate")
    #println("\t\tstate = ",s," agId = ",agID, " oaId = ",oaID)

    lvl = level(p.problem)
    pomcpsolver = p.solver.solvers[lvl+1][1]

    local aj::oaction_type(p.problem)
    if typeof(mj) <: Intentional_Model
        #println("\t\tLevel $lvl, mj is intentional")
        b_j = mj.belief
        mj_frame = mj.frame
        if isnull(b_j.act_prob)
            #println("\t\tb_j.actProb is null")
            j_planner = solve(p.solver, mj_frame)
            b_j.act_prob = Nullable(actionProb(j_planner, b_j))
        end
        aj_prob = get(b_j.act_prob)
        aj = rand(aj_prob,p.rng)
    else
        #println("\t\t Level $lvl, mj is sub-intentional")
        frame_j = mj.frame
        hist_j = mj.history
        j_solver = solver(frame_j, rng=p.rng) #NOTE: Other Subintentional models should implement the same function calls
        j_planner = solve(j_solver, frame_j)
        aj = action(j_planner, hist_j)
    end
    #println("\t\ta_$oaID = $aj")

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
            criterion_value = t.v[node] + pomcpsolver.c*sqrt(ltn/n)
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

    sp, o, r = generate_sor(p.problem, s, a, aj, mj.frame, p.rng)

    hao = get(t.o_lookup, (ha, o), 0)
    if hao == 0
        hao = insert_obs_node!(t, p.problem, ha, o)
        v = BasicPOMCP.estimate_value(p.solved_estimator,
                           p.problem.thisPOMDP,
                           sp,
                           BasicPOMCP.POMCPObsNode(t, hao),
                           steps-1)
        R = r + discount(p.problem)*v
    else
        mjp = rand(update_model(p.problem, mj, s, a, aj, sp, p.rng,p.solver), p.rng)
        isp = InteractiveState(sp, mjp)
        R = r + discount(p.problem)*simulate(p, isp, BasicPOMCP.POMCPObsNode(t, hao), steps-1)
    end

    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]

    return R
end
