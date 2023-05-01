from __future__ import absolute_import

import drl_lib.to_do.dynamic_programming as dynamic_programming
import drl_lib.to_do.monte_carlo_methods as monte_carlo_methods
from TicTacToe import TicTacToe


def p_line_world(s, a, s_p, r):
    assert (0 <= s <= 4)
    assert (0 <= s_p <= 4)
    assert (0 <= a <= 1)
    assert (0 <= r <= 2)
    if s == 0 or s == 4:
        return 0.0
    if s + 1 == s_p and a == 1 and r == 1 and s != 3:
        return 1.0
    if s + 1 == s_p and a == 1 and r == 2 and s == 3:
        return 1.0
    if s - 1 == s_p and a == 0 and r == 1 and s != 1:
        return 1.0
    if s - 1 == s_p and a == 0 and r == 0 and s == 1:
        return 1.0
    return 0.0


def pi_random_line_world(s, a):
    if s == 0 or s == 6:
        return 0.0
    return 0.5


if __name__ == "__main__":
    S_Line_World = [0, 1, 2, 3, 4]
    A_Line_World = [0, 1]  # Gauche, Droite
    R_Line_World = [-1.0, 0.0, 1.0]

    # print(dynamic_programming.policy_evaluation_on_line_world(S_Line_World,
    #                                                           A_Line_World,
    #                                                           R_Line_World,
    #                                                           p_line_world,
    #                                                           pi_random_line_world))
    # print(dynamic_programming.policy_iteration_on_line_world(S_Line_World,
    #                                                          A_Line_World,
    #                                                          R_Line_World,
    #                                                          p_line_world))

    print(dynamic_programming.value_iteration_on_line_world(S_Line_World,
                                                            A_Line_World,
                                                            R_Line_World,
                                                            p_line_world))

    # tic = TicTacToe()
    # tic.moves
    # dynamic_programming.demo()
    # monte_carlo_methods.demo()
    # temporal_difference_learning.demo()
