from __future__ import absolute_import

import drl_lib.to_do.dynamic_programming as dynamic_programming
import drl_lib.to_do.monte_carlo_methods as monte_carlo_methods
from TicTacToe import TicTacToe


def transition(state, action):
    """
    Calcule la probabilité de transition pour un état et une action donnés dans un environnement de type grid world de 5x5.

    Args:
        state (tuple): Un tuple (x,y) représentant les coordonnées de l'état actuel.
        action (int): Un entier représentant l'action à prendre. 0: haut, 1: droite, 2: bas, 3: gauche.

    Returns:
        next_states (list): Une liste de tuples (state, proba, reward, done) représentant les états suivants possibles,
        ainsi que leur probabilité de transition, leur récompense associée et un booléen indiquant si l'état suivant est un état terminal.
    """

    # Initialisation de la liste des états suivants possibles
    next_states = []

    # Récupération des coordonnées de l'état actuel
    x, y = state

    # Calcul des coordonnées de l'état suivant en fonction de l'action choisie
    if action == 0:  # haut
        next_x, next_y = x, y + 1
    elif action == 1:  # droite
        next_x, next_y = x + 1, y
    elif action == 2:  # bas
        next_x, next_y = x, y - 1
    elif action == 3:  # gauche
        next_x, next_y = x - 1, y

    # Vérification que l'état suivant est dans la grille
    if next_x < 0 or next_x > 4 or next_y < 0 or next_y > 4:
        next_x, next_y = x, y

    # Calcul de la probabilité de transition pour chaque état suivant possible
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i != 0 and j != 0:
                continue
            if next_x + i < 0 or next_x + i > 4 or next_y + j < 0 or next_y + j > 4:
                continue
            if i == 0 and j == 0:
                continue
            if (next_x + i, next_y + j) == (2, 2):
                next_states.append(((2, 2), 1 / 3, 0, True))
            else:
                next_states.append(((next_x + i, next_y + j), 1 / 3, -1, False))

    return next_states


if __name__ == "__main__":
    # print(max(1, 2 ))
    # tic = TicTacToe()
    # tic.moves
    dynamic_programming.demo()
    # monte_carlo_methods.demo()
    # temporal_difference_learning.demo()
    # print(dynamic_programming.p_grid_world(0, 1, 1, 1))
