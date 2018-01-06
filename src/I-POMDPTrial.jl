function tester()
    lvl_0_tiger_pomdp = IPOMCP.Level_0_tigerPOMDP()
    lvl_l_tiger_pomdp = IPOMCP.Level_l_tigerPOMDP()

    tiger_static_dist_frame_sets = IPOMCP.initialize_static_distribution_frame_sets(lvl_l_tiger_pomdp)
    tiger_intentional_frame_sets = IPOMCP.initialize_intentional_frame_sets(lvl_l_tiger_pomdp)

    tiger_ipomdp = IPOMCP.IPOMDP_2(1,1,lvl_l_tiger_pomdp, tiger_static_dist_frame_sets, tiger_intentional_frame_sets)
    d = IPOMCP.Tiger_Frame_Distribution(tiger_ipomdp)

    ipomcp_solver = IPOMCP.IPOMCPSolver([(POMCPSolver(max_depth=5,tree_queries=100),100,3.0),(POMCPSolver(max_depth=5,tree_queries=100),200,6.0),(POMCPSolver(max_depth=5,tree_queries=100),500,9.0)])

    nested_init_bel = IPOMCP.rand(MersenneTwister(2), d, [20,20])

    up = IPOMCP.SimpleInteractiveParticleFilter(tiger_ipomdp,LowVarianceResampler(ipomcp_solver.solvers[2][2]),Base.GLOBAL_RNG,ipomcp_solver)

    nested_next_bel = IPOMCP.update(up, nested_init_bel, 2, (1,2))

    ipomcp_planner = solve(ipomcp_solver, tiger_ipomdp)

    a = IPOMCP.action(ipomcp_planner,nested_init_bel)
    println("Action = $a")
end
