function tester()
    num_iter = 10
    total_planning_time = 0.0
    total_updating_time = 0.0

    lvl_l_tiger_pomdp_i = IPOMCP.Level_l_tigerPOMDP()
    lvl_l_tiger_pomdp_j = IPOMCP.Level_l_tigerPOMDP()

    tiger_static_dist_frame_sets_i = IPOMCP.initialize_static_distribution_frame_sets(lvl_l_tiger_pomdp_i)
    tiger_intentional_frame_sets_i = IPOMCP.initialize_intentional_frame_sets(lvl_l_tiger_pomdp_i)
    tiger_static_dist_frame_sets_j = IPOMCP.initialize_static_distribution_frame_sets(lvl_l_tiger_pomdp_j)
    tiger_intentional_frame_sets_j = IPOMCP.initialize_intentional_frame_sets(lvl_l_tiger_pomdp_j)

    tiger_ipomdp_i = IPOMCP.IPOMDP_2(1,1,lvl_l_tiger_pomdp_i, tiger_static_dist_frame_sets_i, tiger_intentional_frame_sets_i)
    tiger_ipomdp_j = IPOMCP.IPOMDP_2(2,1,lvl_l_tiger_pomdp_j, tiger_static_dist_frame_sets_j, tiger_intentional_frame_sets_j)
    dist_i = initial_state_distribution(tiger_ipomdp_i)
    dist_j = initial_state_distribution(tiger_ipomdp_j)
    num_particles_i = num_nested_particles(tiger_ipomdp_i.thisPOMDP, tiger_ipomdp_i)
    num_particles_j = num_nested_particles(tiger_ipomdp_j.thisPOMDP, tiger_ipomdp_j)
    for itr in 1:num_iter
        println("************Iteration $itr**************")
        rng = MersenneTwister(itr)
        init_belief_i =  rand(rng, dist_i, num_particles_i)
        init_belief_j =  rand(rng, dist_j, num_particles_j)

        #=println("Initial Belief for i:")
        print(init_belief_i)
        println()
        println("Physical state belief:", get_physical_state_probability(init_belief_i),"\n\n")

        println("Initial Belief for j:")
        print(init_belief_j)
        println()
        println("Physical state belief:", get_physical_state_probability(init_belief_j),"\n\n")=#

        ipomcp_solver_i = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=5,tree_queries=100,rng=rng),100,5.0),
                                (POMCPSolver(max_depth=5,tree_queries=500,rng=rng),300,30.0)])
        ipomcp_solver_j = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=5,tree_queries=100,rng=rng),100,5.0),
                                (POMCPSolver(max_depth=5,tree_queries=500,rng=rng),300,30.0)])

        ipomcp_planner_i = solve(ipomcp_solver_i, tiger_ipomdp_i)
        ipomcp_planner_j = solve(ipomcp_solver_j, tiger_ipomdp_j)
        init_state = rand(rng, initial_state_distribution(tiger_ipomdp_i.thisPOMDP))
        println("s = $init_state")
        println("computing...")
        updater_i = updater(ipomcp_planner_i)
        updater_j = updater(ipomcp_planner_j)
        t1 = time_ns()
        hr = HistoryRecorder(max_steps = 15, rng = rng)
        hist_i, hist_j = simulate(hr, tiger_ipomdp_i, ipomcp_planner_i, tiger_ipomdp_j, ipomcp_planner_j)
        t2 = time_ns()
        planningTime = (t2 - t1)/1.0e9
        println("planning time: ",planningTime)
        total_planning_time += planningTime
    end
    println("Average Planning Time: ", total_planning_time/(num_iter*2))
    #println("Average Belief Update Time: ", total_updating_time/(num_iter*2))
    #=actionCounts = zeros(Int64,3)
    for i in 1:10
        rng = MersenneTwister(i)
        ipomcp_solver = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=5,tree_queries=100,rng=rng),100,5.0),
                                (POMCPSolver(max_depth=5,tree_queries=500,rng=rng),300,30.0),
                                (POMCPSolver(max_depth=10,tree_queries=900, rng=rng),500,9.0)])
        #println("IPOMDP Solver\n", ipomcp_solver)

        nested_init_bel = IPOMCP.rand(rng, d, [100,200])
        #print(nested_init_bel,0)
        print(get_physical_state_probability(nested_init_bel))
        print("\n\t")
        #up = IPOMCP.SimpleInteractiveParticleFilter(tiger_ipomdp,LowVarianceResampler(ipomcp_solver.solvers[2][2]),rng,ipomcp_solver)

        #nested_next_bel = IPOMCP.update(up, nested_init_bel, 2, (1,2))
        #print(get_physical_state_probability(nested_next_bel))
        #println("\n\n\n")

        ipomcp_planner = solve(ipomcp_solver, tiger_ipomdp)

        a = IPOMCP.action(ipomcp_planner,nested_init_bel)
        println("Action = $a")
        actionCounts[a] += 1
    end=#

    #println(actionCounts)
end

function test_intersection_problem()
    num_iter = 1
    total_planning_time = 0.0
    total_updating_time = 0.0

    pomdp_i = IPOMCP.IntersectionPOMDP(agID = 1)
    pomdp_j = IPOMCP.IntersectionPOMDP(agID = 2)

    static_dist_frame_sets_i =
        IPOMCP.initialize_static_distribution_frame_sets(pomdp_i)
    intentional_frame_sets_i =
        IPOMCP.initialize_intentional_frame_sets(pomdp_i)
    static_dist_frame_sets_j =
        IPOMCP.initialize_static_distribution_frame_sets(pomdp_j)
    intentional_frame_sets_j =
        IPOMCP.initialize_intentional_frame_sets(pomdp_j)

    ipomdp_i = IPOMCP.IPOMDP_2(1,1,pomdp_i, static_dist_frame_sets_i, intentional_frame_sets_i)
    ipomdp_j = IPOMCP.IPOMDP_2(2,1,pomdp_j, static_dist_frame_sets_j, intentional_frame_sets_j)
    #println(ipomdp_i)
    #println(ipomdp_j)

    dist_i = initial_state_distribution(ipomdp_i)
    dist_j = initial_state_distribution(ipomdp_j)
    #println(dist_i)
    #println(dist_j)
    num_particles_i = num_nested_particles(ipomdp_i.thisPOMDP, ipomdp_i)
    num_particles_j = num_nested_particles(ipomdp_j.thisPOMDP, ipomdp_j)

    for itr in 1:num_iter
        println("************Iteration $itr**************")
        rng = MersenneTwister(itr)
        init_belief_i =  rand(rng, dist_i, num_particles_i)
        init_belief_j =  rand(rng, dist_j, num_particles_j)

        #=println("Initial Belief for i:")
        print(init_belief_i)
        println()
        #println("Physical state belief:", get_physical_state_probability(init_belief_i),"\n\n")

        println("Initial Belief for j:")
        print(init_belief_j)
        println()
        #println("Physical state belief:", get_physical_state_probability(init_belief_j),"\n\n")
        =#
        ipomcp_solver_i = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=5,tree_queries=100,rng=rng),100,5.0),
                                (POMCPSolver(max_depth=5,tree_queries=500,rng=rng),300,30.0)])
        ipomcp_solver_j = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=5,tree_queries=100,rng=rng),100,5.0),
                                (POMCPSolver(max_depth=5,tree_queries=500,rng=rng),300,30.0)])

        ipomcp_planner_i = solve(ipomcp_solver_i, ipomdp_i)
        ipomcp_planner_j = solve(ipomcp_solver_j, ipomdp_j)
        init_state = rand(rng, initial_state_distribution(ipomdp_i.thisPOMDP))
        println("s = $init_state")
        println("computing...")
        updater_i = updater(ipomcp_planner_i)
        updater_j = updater(ipomcp_planner_j)
        t1 = time_ns()
        hr = HistoryRecorder(max_steps = 15, rng = rng)
        hist_i, hist_j = simulate(hr, ipomdp_i, ipomcp_planner_i, ipomdp_j, ipomcp_planner_j)
        t2 = time_ns()
        planningTime = (t2 - t1)/1.0e9
        println("planning time: ",planningTime)
        total_planning_time += planningTime
    end
    println("Average Planning Time: ", total_planning_time/(num_iter*2))
end
