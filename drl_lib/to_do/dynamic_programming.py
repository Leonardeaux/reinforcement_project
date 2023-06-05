import numpy as np

from drl_lib.do_not_touch.mdp_env_wrapper import Env1
from drl_lib.do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction

S_Line_World = [0, 1, 2, 3, 4, 5, 6]
A_Line_World = [0, 1]  # Gauche, Droite
R_Line_World = [-1.0, 0.0, 1.0]

S_Grid_World_1 = np.array([[0, 1, 2, 3, 4],
                           [5, 6, 7, 8, 9],
                           [10, 11, 12, 13, 14],
                           [15, 16, 17, 18, 19],
                           [20, 21, 22, 23, 24]])
S_Grid_World = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
A_Grid_World = [0, 1, 2, 3]  # Haut, Bas, Gauche, Droite
R_Grid_World = [-1.0, 0.0, 1.0]


def p_line_world(s, a, s_p, r):
    assert (0 <= s <= 6)
    assert (0 <= s_p <= 6)
    assert (0 <= a <= 1)
    assert (0 <= r <= 2)
    if s == 0 or s == 6:
        return 0.0
    if s + 1 == s_p and a == 1 and r == 1 and s != 5:
        return 1.0
    if s + 1 == s_p and a == 1 and r == 2 and s == 5:
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


def p_grid_world(s, a, s_p, r):
    assert (0 <= s <= 24)
    assert (0 <= s_p <= 24)
    assert (0 <= a <= 3)
    assert (0 <= r <= 2)

    if s == 4 or s == 24:
        return 0.0
    if a == 0 and s - 5 < 0:
        return 0.0
    if a == 1 and s + 5 > 24:
        return 0.0
    if a == 2 and s % 5 == 0:
        return 0.0
    if a == 3 and s % 5 == 4:
        return 0.0
    if s + 1 == s_p and a == 3 and r == 2 and s == 23:
        return 1.0
    if s + 5 == s_p and a == 1 and r == 2 and s == 19:
        return 1.0
    if s + 1 == s_p and a == 3 and r == 0 and s == 3:
        return 1.0
    if s - 5 == s_p and a == 0 and r == 0 and s == 9:
        return 1.0
    if s + 1 == s_p and a != 3:
        return 0.0
    if s - 1 == s_p and a != 2:
        return 0.0
    if s + 5 == s_p and a != 1:
        return 0.0
    if s - 5 == s_p and a != 0:
        return 0.0
    if r == 1:
        return 1.0
    return 0.0


def pi_random_grid_world(s, a):
    if s == 4 or s == 24:
        return 0.0
    return 0.5


def policy_evaluation_on_line_world(S, A, R, p, pi, theta: float = 0.0000001) -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    V = {s: 0 for s in S}

    while True:
        delta = 0.0

        for s in S:
            old_v = V[s]
            total = 0.0
            for a in A:
                total_inter = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total_inter += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                total_inter = pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return V


def policy_iteration_on_line_world(S, A, R, p, theta: float = 0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    pi = {s: np.random.choice(A) for s in S}
    V = {s: 0 for s in S}

    while True:
        # Policy Evaluation
        while True:
            delta = 0.0
            for s in S:
                old_v = V[s]
                total = 0.0
                a = pi[s]
                for s_p in S:
                    for r in range(len(R)):
                        total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                V[s] = total
                delta = max(delta, np.abs(V[s] - old_v))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in S:
            old_a = pi[s]

            # argmax from scratch
            best_a = None
            best_a_score = None
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                if best_a is None or total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s] = best_a
            if old_a != best_a:
                policy_stable = False
        if policy_stable:
            break
    return PolicyAndValueFunction(pi, V)


def value_iteration_on_line_world(S, A, R, p, theta: float = 0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = {s: 0 for s in S}
    pi = {s: 0 for s in S}

    while True:
        delta = 0.0

        for s in S:
            old_v = V[s]
            total = 0.0
            for a in A:
                total_inter = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total_inter += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])

                total = max(total, total_inter)

                if V[s] < total_inter:
                    pi[s] = A[a]

            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))

        if delta < theta:
            break

    return PolicyAndValueFunction(pi, V)


def policy_evaluation_on_grid_world(S, A, R, p, pi, theta: float = 0.0000001) -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    V = {s: 0 for s in S}
    while True:
        delta = 0.0

        for s in S:
            old_v = V[s]
            total = 0.0
            for a in A:
                total_inter = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total_inter += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                total_inter = pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return V


def policy_iteration_on_grid_world(S, A, R, p, theta: float = 0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    pi = {s: np.random.choice(A) for s in S}
    V = {s: 0 for s in S}

    while True:
        # Policy Evaluation
        while True:
            delta = 0.0
            for s in S:
                old_v = V[s]
                total = 0.0
                a = pi[s]
                for s_p in S:
                    for r in range(len(R)):
                        total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                V[s] = total
                delta = max(delta, np.abs(V[s] - old_v))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in S:
            old_a = pi[s]

            # argmax from scratch
            best_a = None
            best_a_score = None
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                if best_a is None or total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s] = best_a
            if old_a != best_a:
                policy_stable = False
        if policy_stable:
            break
    return PolicyAndValueFunction(pi, V)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    # TODO
    pass


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def demo():
    print(policy_evaluation_on_line_world(S_Line_World,
                                          A_Line_World,
                                          R_Line_World,
                                          p_line_world,
                                          pi_random_line_world))
    print(policy_iteration_on_line_world(S_Line_World,
                                         A_Line_World,
                                         R_Line_World,
                                         p_line_world))
    print(value_iteration_on_line_world(S_Line_World,
                                        A_Line_World,
                                        R_Line_World,
                                        p_line_world))

    print(policy_evaluation_on_grid_world(S_Grid_World,
                                          A_Grid_World,
                                          R_Grid_World,
                                          p_grid_world,
                                          pi_random_grid_world))
    print(policy_iteration_on_grid_world(S_Grid_World,
                                         A_Grid_World,
                                         R_Grid_World,
                                         p_grid_world))
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
