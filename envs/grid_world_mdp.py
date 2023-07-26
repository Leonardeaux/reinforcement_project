from drl_lib.do_not_touch.contracts import MDPEnv
import numpy as np


class GridWorldEnv(MDPEnv):
    def __init__(self, rows: int, cols: int, win_coords: (int, int), lost_coords: (int, int)):
        assert (rows > 1 and cols > 1)
        self.rows = rows
        self.cols = cols
        self.cells_nb = self.rows * self.cols
        self.S = np.arange(self.cells_nb)
        self.A = np.array([0, 1, 2, 3], dtype=int)  # Haut, Droite, Bas, Gauche
        self.R = np.array([-1, 0, 1], dtype=int)
        self.p_matrix = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self.pi_random = np.full((len(self.S), len(self.A)), 1 / len(self.A))
        self.win_coords = win_coords
        self.loose_coords = lost_coords

        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) != self.win_coords and (i, j) != self.loose_coords:
                    if i > 0:  # Up
                        self.p_matrix[i * self.cols + j, 0, (i - 1) * self.cols + j, 1] = 1.0
                    if j < self.cols - 1:  # Right
                        self.p_matrix[i * self.cols + j, 1, i * self.cols + j + 1, 1] = 1.0
                    if i < self.rows - 1:  # Down
                        self.p_matrix[i * self.cols + j, 2, (i + 1) * self.cols + j, 1] = 1.0
                    if j > 0:  # Left
                        self.p_matrix[i * self.cols + j, 3, i * self.cols + j - 1, 1] = 1.0

        # Add terminal states and rewards
        self.p_matrix[self.win_coords[0] * self.cols + self.win_coords[1], :,
        self.win_coords[0] * self.cols + self.win_coords[1], 2] = 1.0
        self.p_matrix[self.loose_coords[0] * self.cols + self.loose_coords[1], :,
        self.loose_coords[0] * self.cols + self.loose_coords[1], 0] = 1.0

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        if s == self.win_coords[0] * self.cells_nb + self.win_coords[1] or \
                s == self.loose_coords[0] * self.cells_nb + self.loose_coords[1]:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: int) -> float:
        return self.p_matrix[s, a, s_p, r]

    def view_state(self, s: int):
        pass
