import numpy as np
from drl_lib.do_not_touch.contracts import MDPEnv
from drl_lib.do_not_touch.result_structures import PolicyAndValueFunction


def value_iteration(environment: MDPEnv,
                    gamma: float = 0.99999,
                    theta: float = 0.0000001) -> PolicyAndValueFunction:
    S = environment.states()
    A = environment.actions()
    R = environment.rewards()
    V = {s: 0 for s in S}
    p = environment.transition_probability
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
                        total_inter += p(s, a, s_p, r) * (R[r] + gamma * V[s_p])

                total = max(total, total_inter)

                if V[s] < total_inter:
                    pi[s] = A[a]

            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))

        if delta < theta:
            break

    return PolicyAndValueFunction(pi, V)
