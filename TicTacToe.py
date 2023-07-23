import numpy as np
from drl_lib.do_not_touch.contracts import SingleAgentEnv


class TicTacToe(SingleAgentEnv):

    # initialisation du board de 9 cases et current player a 1
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1

    def state_id(self):
        return str(self.board.astype(int))

    def is_game_over(self):
        return self.check_winner() or np.all(self.board != 0)

    def act_with_action_id(self, action_id):
        self.board[action_id] = self.current_player
        self.current_player = -self.current_player

    def score(self):
        winner = self.check_winner()
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        else:
            return 0

    def available_actions_ids(self):
        return [i for i, val in enumerate(self.board) if val == 0]

    def reset(self):
        pass

    def view(self):
        pass

    def reset_random(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1

    def reset_to_state(self, state):
        self.board = np.array([int(float(s)) for s in state.strip('[]').split(',')])

    def action_space(self):
        return self.available_actions_ids()

    def check_winner(self):
        winning_moves = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        return next(
            (
                self.board[moves[0]]
                for moves in winning_moves
                if self.board[moves[0]] == self.board[moves[1]] == self.board[moves[2]] != 0
            ),
            None,
        )
