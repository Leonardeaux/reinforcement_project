import numpy as np
from tqdm import tqdm
from drl_lib.do_not_touch.contracts import SingleAgentEnv
from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction


def init(env, s, pi, q):
    actions = env.available_actions_ids()

    if s not in pi:
        pi[s] = {}
        q[s] = {}
        for a in actions:
            pi[s][a] = 1.0 / len(actions)
            q[s][a] = 0.0
    return pi, q


def epsilon_func(env, s, epsilon, q):
    actions = env.available_actions_ids()

    prob = np.full(len(actions), epsilon / len(actions))

    best = np.argmax([q[s][a] for a in actions])
    prob[best] += (1.0 - epsilon)

    return np.random.choice(actions, p=prob)


def sarsa_for_grid_world(env: SingleAgentEnv, alpha: float, epsilon: float, discount: float,
                         iterations: int) -> PolicyAndActionValueFunction:
    assert (epsilon > 0)

    pi = {}
    q = {}

    for i in tqdm(range(iterations)):
        env.reset()
        state = env.state_id()
        pi, q = init(env, state, pi, q)

        action = epsilon_func(env, state, epsilon, q)
        while not env.is_game_over():
            old_state = state
            old_action = action
            score = env.score()
            env.act_with_action_id(action)
            r = env.score() - score

            state = env.state_id()
            pi, q = init(env, state, pi, q)
            if not env.is_game_over():
                action = epsilon_func(env, state, epsilon, q)

            q[old_state][old_action] += alpha * (
                        r + discount * (q[state][action] if action is not None else 0) - q[old_state][old_action])

    for state in q.keys():
        best = max(q[state], key=q[state].get)
        for action in pi[state].keys():
            pi[state][action] = float(action == best)

    return PolicyAndActionValueFunction(pi, q)
