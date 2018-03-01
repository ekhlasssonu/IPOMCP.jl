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

        ipomcp_solver_i = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=4,tree_queries=100,rng=rng,c=5.0),num_particles_i[1],5.0),
                                (POMCPSolver(max_depth=4,tree_queries=1000,rng=rng,c=5.0),num_particles_i[2],30.0)])
        ipomcp_solver_j = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=4,tree_queries=100,rng=rng,c=5.0),num_particles_j[1],5.0),
                                (POMCPSolver(max_depth=4,tree_queries=1000,rng=rng,c=5.0),num_particles_j[2],30.0)])

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

        #Verification print:
        belief_hist_i = belief_hist(hist_i)
        belief_hist_j = belief_hist(hist_j)

        state_hist_i = state_hist(hist_i)
        state_hist_j = state_hist(hist_j)

        action_hist_i = action_hist(hist_i)
        action_hist_j = action_hist(hist_j)

        obs_hist_i = observation_hist(hist_i)
        obs_hist_j = observation_hist(hist_j)

        rwd_hist_i = reward_hist(hist_i)
        rwd_hist_j = reward_hist(hist_j)

        for step in 1:length(belief_hist_i)-1
            println("Step: ", step-1)
            state_belief_i = get_physical_state_probability(belief_hist_i[step])
            state_belief_j = get_physical_state_probability(belief_hist_j[step])

            print("\tAgent_i: ")
            sparse_print(belief_hist_i[step])
            println(" Action_i = ",action_hist_i[step], " Reward_i = ", rwd_hist_i[step]);
            print("\tAgent_j: ")
            sparse_print(belief_hist_j[step])
            println(" Action_j = ",action_hist_j[step], " Reward_j = ", rwd_hist_j[step]);
            println()
        end
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

function test_intersection_problem(;num_iter = 100, vel_dev_cost = -5.0, hard_brake_cost = -5.0, collision_cost = -500.0,
    success_reward = 100.0, l0_max_depth = 5, l1_max_depth = 5, l0_queries = 200, l1_queries = 1000, l0_lambda=0.5, l1_lambda = 5.0, max_steps = 20)


    pomdp_i = IPOMCP.IntersectionPOMDP(agID = 1, decision_timestep=0.5,
                        vel_dev_cost=vel_dev_cost,hard_brake_cost=-5.0,
                        collision_cost=-500.0,success_reward=100.0)
    pomdp_j = IPOMCP.IntersectionPOMDP(agID = 2, decision_timestep=0.5,
                        vel_dev_cost=vel_dev_cost,hard_brake_cost=-5.0,
                        collision_cost=-500.0,success_reward=100.0)

    static_dist_frame_sets_i =
        IPOMCP.initialize_static_distribution_frame_sets(pomdp_i)
    intentional_frame_sets_i =
        IPOMCP.initialize_intentional_frame_sets([pomdp_i,pomdp_j])
    static_dist_frame_sets_j =
        IPOMCP.initialize_static_distribution_frame_sets(pomdp_j)
    intentional_frame_sets_j =
        IPOMCP.initialize_intentional_frame_sets([pomdp_i,pomdp_j])

    for lvl_j in 0:1
        println("************Lvl_j = $lvl_j*****************")
        for lvl_i in 0:lvl_j
            println("************Lvl_i = $lvl_i*****************")
            ipomdp_i = IPOMCP.IPOMDP_2(1,lvl_i,pomdp_i, static_dist_frame_sets_i, intentional_frame_sets_i)
            ipomdp_j = IPOMCP.IPOMDP_2(2,lvl_j,pomdp_j, static_dist_frame_sets_j, intentional_frame_sets_j)
            println(ipomdp_i)
            println(ipomdp_j)

            dist_i = initial_state_distribution(ipomdp_i)
            dist_j = initial_state_distribution(ipomdp_j)
            #println(dist_i)
            #println(dist_j)
            num_particles_i = num_nested_particles(ipomdp_i.thisPOMDP, ipomdp_i)
            num_particles_j = num_nested_particles(ipomdp_j.thisPOMDP, ipomdp_j)

            total_planning_time = 0.0
            total_updating_time = 0.0
            num_success = 0
            num_steps = 0
            avg_rwd_i = 0.0
            avg_rwd_j = 0.0
            num_hardbrakes_i = 0
            num_hardbrakes_j = 0
            for itr in 1:num_iter
                println("************Iteration $itr**************")
                rng = MersenneTwister(itr)
                rng_i = MersenneTwister(itr*5 + 3)
                rng_j = MersenneTwister(itr*3 + 5)
                ipomcp_solver_i = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=l0_max_depth,tree_queries=l0_queries,c=1.0,rng=rng_i),num_particles_i[1],l0_lambda),
                                        (POMCPSolver(max_depth=l1_max_depth,tree_queries=l1_queries,c=1.0,rng=rng_i),num_particles_i[2],l1_lambda)])
                ipomcp_solver_j = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=l0_max_depth,c=1.0,tree_queries=l0_queries,rng=rng_j),num_particles_j[1],l0_lambda),
                                        (POMCPSolver(max_depth=l1_max_depth,tree_queries=l1_queries,c=1.0,rng=rng_j),num_particles_j[2],l1_lambda)])

                ipomcp_planner_i = solve(ipomcp_solver_i, ipomdp_i)
                ipomcp_planner_j = solve(ipomcp_solver_j, ipomdp_j)
                #init_state = rand(rng, initial_state_distribution(ipomdp_i.thisPOMDP))
                #println("s = $init_state")
                #println("computing...")
                t1 = time_ns()
                hr = HistoryRecorder(max_steps = max_steps, rng = rng)
                hist_i, hist_j = simulate(hr, ipomdp_i, ipomcp_planner_i, ipomdp_j, ipomcp_planner_j)
                t2 = time_ns()
                planningTime = (t2 - t1)/1.0e9
                #println("planning time: ",planningTime)
                total_planning_time += planningTime

                #Verification print:
                belief_hist_i = belief_hist(hist_i)
                belief_hist_j = belief_hist(hist_j)

                state_hist_i = state_hist(hist_i)
                state_hist_j = state_hist(hist_j)

                action_hist_i = action_hist(hist_i)
                action_hist_j = action_hist(hist_j)

                obs_hist_i = observation_hist(hist_i)
                obs_hist_j = observation_hist(hist_j)

                rwd_hist_i = reward_hist(hist_i)
                rwd_hist_j = reward_hist(hist_j)

                total_rwd_i = 0.0
                total_rwd_j = 0.0

                if state_hist_i[length(state_hist_i)].terminal == 2
                    num_success += 1;
                    print("Success. ")
                    num_steps += length(belief_hist_i)-1
                elseif state_hist_i[length(state_hist_i)].terminal == 1
                    print("Collision. ")
                else
                    print("Failure. ")
                end

                for step in 1:length(belief_hist_i)-1
                    a_i = VehicleActionSpace_Intersection().actions[action_hist_i[step]]
                    if (a_i.accl <= -4.0)
                        num_hardbrakes_i += 1
                    end

                    a_j = VehicleActionSpace_Intersection().actions[action_hist_j[step]]
                    if (a_j.accl <= -4.0)
                        num_hardbrakes_j += 1
                    end

                    #sparse_print(belief_hist_i[step])
                    if state_hist_i[length(state_hist_i)].terminal != 2
                        println("Step: ", step-1)
                        println("State: ",state_hist_i[step])
                        print("\tAgent_i: ")
                        println(" Action_i = ",action_hist_i[step], " Reward_i = ", rwd_hist_i[step]);
                        print("\tAgent_j: ")
                        #sparse_print(belief_hist_j[step])
                        println(" Action_j = ",action_hist_j[step], " Reward_j = ", rwd_hist_j[step]);
                    end
                    total_rwd_i += rwd_hist_i[step]
                    total_rwd_j += rwd_hist_j[step]
                    #println()
                end
                println("Cum_rwd_i = $total_rwd_i, Cum_rwd_j = $total_rwd_j ")
                avg_rwd_i += total_rwd_i
                avg_rwd_j += total_rwd_j
            end
            println("Average Planning Time: ", total_planning_time/(num_iter*2))
            println("Num success = $num_success avg num time steps = ", 1.0 * num_steps/num_success)
            println("Avg num hardbrakes_i = ", 1.0*num_hardbrakes_i/num_iter, " Avg num hardbrakes_j = ", 1.0*num_hardbrakes_j/num_iter)
            println("Avg rwd_i = ", avg_rwd_i/num_iter, " Avg rwd_j = ", avg_rwd_j/num_iter)
        end
    end
end
