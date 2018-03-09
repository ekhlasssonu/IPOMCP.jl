using IPOMCP
using Base.Test

# write your own tests here
using POMDPs

#tester()

test_intersection_problem(num_iter=100, decision_timestep=1.0, vel_dev_cost = -5.0, hard_brake_cost = -5.0, collision_cost = -500.0,
    success_reward = 100.0, l0_max_depth = 4, l1_max_depth = 4, l0_queries = 200, l1_queries = 500,
    l0_lambda=0.5, l1_lambda = 5.0, max_steps = 35)
@test 1 == 1
