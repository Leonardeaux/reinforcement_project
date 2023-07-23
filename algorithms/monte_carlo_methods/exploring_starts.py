import numpy as np
from collections import defaultdict
from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_lib.do_not_touch.contracts import SingleAgentEnv
from envs.grid_world_for_single_agent import GridWorldSAEnv
from TicTacToe import TicTacToe


def exploring_starts(environment: SingleAgentEnv,
                     gamma: float = 0.99999,
                     max_episodes_count: int = 1000) -> PolicyAndActionValueFunction:
    pi = {}
    q = np.random.uniform(-1.0, 1.0, (environment.state_space(), environment.action_space()))
    q = {i: {j: q[i, j] for j in range(q.shape[1])} for i in range(q.shape[0])}

    returns = [[[] for a in range(environment.action_space())] for s in range(environment.state_space())]

    for _ in range(max_episodes_count):
        environment.reset_random()

        # Création de l'épisode aléatoire
        S = []
        A = []
        R = []
        while not environment.is_game_over():
            state = environment.state_id()
            S.append(state)
            aa = environment.available_actions_ids()

            pi_s = [pi[state, a] for a in aa]

            a = np.random.choice(aa, p=pi_s)

            old_score = environment.score()
            environment.act_with_action_id(a)
            r = environment.score() - old_score
            A.append(a)
            R.append(r)

        G = 0
        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            r_t = R[t]
            G = gamma * G + r_t
            if (s_t, a_t) not in zip(S[: t], A[: t]):
                returns[s_t][a_t].append(G)
                q[s_t, a_t] = np.mean(returns[s_t][a_t])

                pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]

    return PolicyAndActionValueFunction(pi, q)


gw = GridWorldSAEnv(5, 5, 1000, (4, 4), (0, 4))
# t = TicTacToe()
print(exploring_starts(gw).q)
