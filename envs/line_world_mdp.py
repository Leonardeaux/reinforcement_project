from drl_lib.do_not_touch.contracts import MDPEnv
import numpy as np


class LineWorldEnv(MDPEnv):
    def __init__(self, cells_nb: int):
        assert (cells_nb > 2)
        self.cells_nb = cells_nb
        self.S = np.arange(self.cells_nb)
        self.A = np.array([0, 1], dtype=int)  # 0: Left, 1: Right
        self.R = np.array([-1, 0, 1], dtype=int)
        self.p_matrix = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))

        self.pi_random = np.full((len(self.S), len(self.A)), 1 / len(self.A))

        for i in range(1, self.cells_nb - 2):
            self.p_matrix[i, 1, i + 1, 1] = 1.0

        for i in range(2, self.cells_nb - 1):
            self.p_matrix[i, 0, i - 1, 1] = 1.0

        self.p_matrix[cells_nb - 2, 1, cells_nb - 1, 2] = 1.0
        self.p_matrix[1, 0, 0, 0] = 1.0

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        if s == 0 or s == self.cells_nb - 1:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: int) -> float:
        return self.p_matrix[s, a, s_p, r]

    def view_state(self, s: int):
        pass
