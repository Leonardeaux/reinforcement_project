import numpy as np
import random
from collections import defaultdict
from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_lib.do_not_touch.single_agent_env_wrapper import Env2
from TicTacToe import TicTacToe


def generate_state_action_pairs(env):
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

def monte_carlo_es_on_tic_tac_toe_solo(env: TicTacToe,
                                           gamma: float = 0.999999,
                                           max_episodes_count: int = 1000) -> PolicyAndActionValueFunction:
    Q = defaultdict(lambda: defaultdict(lambda: 0))
    Returns = defaultdict(lambda: defaultdict(lambda: []))
    Policy = {}

    all_state_action_pairs = generate_state_action_pairs(env)

    for episode in range(max_episodes_count):
        start_state, start_action, _ = random.choice(all_state_action_pairs)

        env.reset_to_state(start_state)
        player = env.current_player

        episode_list = []
        G = 0

        while not env.is_game_over():
            env.act_with_action_id(start_action)
            reward = env.score()

            episode_list.append((start_state, start_action, reward))
            start_state = env.state_id()

            actions = env.available_actions_ids()

            if len(actions) > 0:
                start_action = random.choice(actions)
                episode_list.append((start_state, start_action, 0))

        for s, a, _ in episode_list:
            Returns[s][a].append(G)
            Q[s][a] = np.average(Returns[s][a])
            if len(env.action_space()) > 0:
                Policy[s] = max(list(range(len(env.action_space()))), key=lambda x: Q[s][x])
            else:
                if env.is_game_over():
                    winner = env.check_winner()
                    if winner:
                        print("winner")
                    else:
                        print("draw")
        episode_list = reversed(episode_list)
    return Policy, Q


env = TicTacToe()
PolicyMC, Q = monte_carlo_es_on_tic_tac_toe_solo(env)
print(PolicyMC)