from __future__ import absolute_import

import numpy as np
import drl_lib.to_do.dynamic_programming as dynamic_programming
import drl_lib.to_do.monte_carlo_methods as monte_carlo_methods
from envs.line_world_for_mdp import LineWorldEnv
from envs.grid_world_for_mdp import GridWorldEnv
from algorithms.dynamic_programming.policy_evaluation import policy_evaluation
from algorithms.dynamic_programming.policy_iteration import policy_iteration
from algorithms.dynamic_programming.value_iteration import value_iteration
from drl_lib.do_not_touch.mdp_env_wrapper import Env1

if __name__ == "__main__":
    pass
    # print(policy_evaluation(line_world, line_world.pi_random))
    # print(policy_iteration(line_world))
    # print(value_iteration(line_world))

    grid_world = GridWorldEnv(5, 5, (4, 4), (0, 4))
    print(policy_evaluation(grid_world, grid_world.pi_random, gamma=0.99, theta=0.01))
    print(policy_iteration(grid_world, gamma=0.99, theta=0.01))
    print(value_iteration(grid_world, gamma=0.99, theta=0.01))
    # env = Env1()
    # pi_random = np.full((len(env.states()), len(env.actions())), 1 / len(env.actions()))
    # print(policy_evaluation(env, pi_random))
    # print(policy_iteration(env))
    # print(value_iteration(env))
