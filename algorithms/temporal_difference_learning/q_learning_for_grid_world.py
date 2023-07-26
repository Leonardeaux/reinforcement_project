import numpy as np
from tqdm import tqdm
from drl_lib.do_not_touch.contracts import SingleAgentEnv
from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction


def q_learning_for_grid_world(env: SingleAgentEnv, lr: float, epsilon: float, discount: float,
                              max_iter: int) -> PolicyAndActionValueFunction:
    assert (epsilon > 0)

    pi = {}
    q = {}
    expl = {}

    for _ in tqdm(range(max_iter)):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            actions = env.available_actions_ids()

            if s not in pi:
                pi[s] = {action: 1.0 / len(actions) for action in actions}
                q[s] = {action: 0.0 for action in actions}
                expl[s] = {action: 1.0 / len(actions) for action in actions}

            optimal_action = max(q[s], key=q[s].get)
            for action_key in expl[s]:
                if action_key == optimal_action:
                    expl[s][action_key] = 1 - epsilon + epsilon / len(actions)
                else:
                    expl[s][action_key] = epsilon / len(actions)

            chosen_action = np.random.choice(list(expl[s].keys()), 1, False, p=list(expl[s].values()))[0]
            score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - score
            s_p = env.state_id()
            next_actions = env.available_actions_ids()

            if env.is_game_over():
                q[s][chosen_action] += lr * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {action: 1.0 / len(next_actions) for action in next_actions}
                    q[s_p] = {action: 0.0 for action in next_actions}
                    expl[s_p] = {action: 1.0 / len(next_actions) for action in next_actions}
                q[s][chosen_action] += lr * (r + discount * max(q[s_p].values()) - q[s][chosen_action])

    for s in q.keys():
        optimal_action_t = max(q[s], key=q[s].get)
        for action_key in q[s]:
            if action_key == optimal_action_t:
                pi[s][action_key] = 1.0
            else:
                pi[s][action_key] = 0.0

    return PolicyAndActionValueFunction(pi, q)
