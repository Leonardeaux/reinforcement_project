import numpy as np
from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_lib.do_not_touch.contracts import SingleAgentEnv
from envs.grid_world_for_single_agent import GridWorldSAEnv
from TicTacToe import TicTacToe


def exploring_starts(environment: SingleAgentEnv,
                     gamma: float = 0.99999,
                     max_episodes_count: int = 1000) -> PolicyAndActionValueFunction:
    pi = {}
    q = {}
    returns = {}

    for _ in range(max_episodes_count):
        environment.reset_random()

        # Création de l'épisode aléatoire
        S = []
        A = []
        R = []
        while not environment.is_game_over():
            s = environment.state_id()
            S.append(s)
            aa = environment.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in aa:
                    pi[s][a] = 1.0 / len(aa)
                    q[s][a] = 0.0
                    returns[s][a] = 0

            chosen_action = np.random.choice(aa, 1, False)[0]

            A.append(chosen_action)
            old_score = environment.score()
            environment.act_with_action_id(chosen_action)
            r = environment.score() - old_score
            R.append(r)
        G = 0

        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            r_t = R[t]
            G = gamma * G + r_t
            if (s_t, a_t) not in zip(S[: t], A[: t]):
                q[s_t][a_t] = (q[s_t][a_t] * returns[s_t][a_t] + G) / (returns[s_t][a_t] + 1)
                returns[s_t][a_t] += 1
                pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]

    return PolicyAndActionValueFunction(pi, q)


import collections
gw = GridWorldSAEnv(5, 5, 1000000, (4, 4), (0, 4))
# t = TicTacToe()
print(collections.OrderedDict(sorted(exploring_starts(gw, gamma=0.99, max_episodes_count=10000).pi.items())))
