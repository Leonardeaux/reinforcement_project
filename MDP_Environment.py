import numpy as np
from drl_lib.do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction


class Environment:
    def __init__(self, height, width, start, goal):
        self.height = height
        self.width = width

        self.R = np.zeros((height, width)) - 1
        self.goal = goal
        self.R[self.goal] = 1.0

        self.S = []
        for i, _ in np.ndenumerate(self.R):
            self.S.append(i)

        self.S = np.asarray(self.S)

        if height > 1:
            self.A = [0, 1, 2, 3]  # Haut, Bas, Gauche, Droite
        else:
            self.A = [0, 1]  # Gauche, Droite

        self.start = list(start)

        self.is_done = False

        # Value function
        self.v = {i: 0 for i in range(height * width)}

    def get_rewards(self):
        return