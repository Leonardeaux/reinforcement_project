import numpy as np
from drl_lib.do_not_touch.contracts import MDPEnv
from drl_lib.do_not_touch.result_structures import PolicyAndValueFunction


def policy_iteration(environment: MDPEnv,
                     gamma: float = 0.99999,
                     theta: float = 0.0000001) -> PolicyAndValueFunction:
    S = environment.states()
    A = environment.actions()
    R = environment.rewards()
    V = {s: 0 for s in S}
    p = environment.transition_probability
    pi = {s: np.random.choice(A) for s in S}

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
                        total += p(s, a, s_p, r) * (R[r] + gamma * V[s_p])
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
                        total += p(s, a, s_p, r) * (R[r] + gamma * V[s_p])
                if best_a is None or total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s] = best_a
            if old_a != best_a:
                policy_stable = False
        if policy_stable:
            break
    return PolicyAndValueFunction(pi, V)
