import numpy as np
from drl_lib.do_not_touch.contracts import SingleAgentEnv


class GridWorldSAEnv(SingleAgentEnv):

    def __init__(self, rows: int, cols: int, max_steps: int, win: (int, int), loose: (int, int)):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.game_over = False
        self.actual_score = 0.0
        self.agent_pos = -1
        self.actual_steps = 0
        self.win = win[0] * self.cols + win[1]
        self.loose = loose[0] * self.cols + loose[1]
        self.actions = [0, 1, 2, 3]  # Haut, Droite, Bas, Gauche

    def state_id(self) -> int:
        return self.agent_pos

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.is_game_over())
        assert (action_id in self.actions)

        x = self.agent_pos % self.cols  # Colonne
        y = self.agent_pos // self.cols  # Ligne

        if action_id == 0:      # Haut
            y -= 1
        elif action_id == 1:    # Droite
            x += 1
        elif action_id == 2:    # bas
            y += 1
        else:                   # Gauche
            x -= 1

        self.agent_pos = y * self.cols + x

        if self.agent_pos == self.loose:
            self.game_over = True
            self.actual_score = -1.0
        elif self.agent_pos == self.win:
            self.game_over = True
            self.actual_score = 1.0

        self.actual_steps += 1
        if self.actual_steps >= self.max_steps:
            self.game_over = True

    def score(self) -> float:
        return self.actual_score

    def available_actions_ids(self) -> np.ndarray:
        if self.is_game_over():
            return np.array([])

        actions = []

        x = self.agent_pos % self.cols
        y = self.agent_pos // self.cols

        if y > 0:               # Haut
            actions.append(0)
        if x < self.cols - 1:   # Droite
            actions.append(1)
        if y < self.rows - 1:   # Bas
            actions.append(2)
        if x > 0:               # Gauche
            actions.append(3)

        return np.array(actions)

    def reset(self):
        self.agent_pos = self.rows // 2 * self.cols + self.cols // 2
        self.game_over = False
        self.actual_steps = 0
        self.actual_score = 0.0

    def view(self):
        pass

    def reset_random(self):
        self.agent_pos = np.random.randint(0, self.rows * self.cols - 1)
        self.game_over = False
        self.actual_steps = 0
        self.actual_score = 0.0

    def action_space(self):
        return len(self.actions)

    def state_space(self):
        return self.rows * self.cols