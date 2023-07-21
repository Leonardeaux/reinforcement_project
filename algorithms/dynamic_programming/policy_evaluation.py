import numpy as np
from drl_lib.do_not_touch.contracts import MDPEnv
from drl_lib.do_not_touch.result_structures import ValueFunction


def policy_evaluation(environment: MDPEnv,
                      pi: np.ndarray,
                      gamma: float = 0.99999,
                      theta: float = 0.0000001) -> ValueFunction:
    S = environment.states()
    A = environment.actions()
    R = environment.rewards()
    V = {s: 0 for s in S}
    p = environment.transition_probability

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
                total_inter = pi[s, a] * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return V
