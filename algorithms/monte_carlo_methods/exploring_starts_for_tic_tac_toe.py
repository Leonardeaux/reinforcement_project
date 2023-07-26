import random
from collections import defaultdict
import numpy as np
from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from envs.tic_tac_toe_single_agent import TicTacToeEnv


def generate_state_action_pairs():
    def simulate_game(states, board, player):
        for action in range(9):  # 9 actions possible
            new_board = board.copy()
            # if the action leads to a valid state
            if new_board[action] == 0:
                new_board[action] = player
                # append the newly created state to the states
                states.append((str(new_board.tolist()), action, player))
                # continue play for the other player
                simulate_game(states, new_board, -player)

    initial_board = np.zeros(9)
    states = []
    simulate_game(states, initial_board, 1)
    return states


def exploring_starts_for_tic_tac_toe(env: TicTacToeEnv,
                                     gamma: float = 0.999999,
                                     max_episodes_count: int = 1000) -> PolicyAndActionValueFunction:
    q = defaultdict(lambda: defaultdict(lambda: 0))
    returns = defaultdict(lambda: defaultdict(lambda: []))
    policy = {}

    all_state_action_pairs = generate_state_action_pairs()

    for episode in range(max_episodes_count):
        start_state, start_action, _ = random.choice(all_state_action_pairs)

        env.reset_to_state(start_state)
        player = env.current_player

        episode_list = []
        G = 0

        while not env.is_game_over():
            env.act_with_action_id(start_action)
            reward = env.score()

            G = gamma * G + reward

            episode_list.append((start_state, start_action, reward))
            start_state = env.state_id()

            actions = env.available_actions_ids()

            if len(actions) > 0:
                start_action = random.choice(actions)
                episode_list.append((start_state, start_action, 0))

        for s, a, _ in episode_list:
            returns[s][a].append(G)
            q[s][a] = np.average(returns[s][a])
            if len(env.action_space()) > 0:
                policy[s] = max(list(range(len(env.action_space()))), key=lambda x: q[s][x])
            else:
                if env.is_game_over():
                    winner = env.check_winner()
                    if winner:
                        env.score()
                    else:
                        env.score()
        episode_list = reversed(episode_list)
    return PolicyAndActionValueFunction(policy, q)
