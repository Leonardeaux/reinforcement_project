from __future__ import absolute_import

import drl_lib.to_do.dynamic_programming as dynamic_programming
import drl_lib.to_do.monte_carlo_methods as monte_carlo_methods
from TicTacToe import TicTacToe


if __name__ == "__main__":
    print(dynamic_programming.p_grid_world(0, 3, 1, 1))
    # dynamic_programming.demo()
    # monte_carlo_methods.demo()
    # temporal_difference_learning.demo()
    # print(dynamic_programming.p_grid_world(0, 1, 1, 1))
