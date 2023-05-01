import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1
        self.winner = None
        self.game_over = False
        self.moves = 0

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1
        self.winner = None
        self.game_over = False
        self.moves = 0

    def get_state(self):
        return self.board

    def get_winner(self):
        return self.winner

    def get_game_over(self):
        return self.game_over

    def get_player(self):
        return self.player

    def get_moves(self):
        return self.moves

    def get_action_space(self):
        pass

    def get_state_space(self):
        pass

    def make_move(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.player
            self.moves += 1
            if self.check_winner():
                self.winner = self.player
                self.game_over = True
            elif self.moves == 9:
                self.game_over = True
            else:
                self.player = -self.player
        else:
            print("Invalid move")

    def check_winner(self):
        # Check rows
        for row in self.board:
            if sum(row) == 3 or sum(row) == -3:
                return True

        # Check columns
        for col in self.board.T:
            if sum(col) == 3 or sum(col) == -3:
                return True

        # Check diagonals
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == 3 or self.board[0][0] + self.board[1][1] + \
                self.board[2][2] == -3:
            return True
        if self.board[0][2] + self.board[1][1] + self.board[2][0] == 3 or self.board[0][2] + self.board[1][1] + \
                self.board[2][0] == -3:
            return True

        return False

    def print_board(self):
        print(self.board)

    def play(self, row, col):
        self.make_move(row, col)
        if self.game_over:
            if self.winner == 1:
                print("Player 1 wins")
            if self.winner == -1:
                print("Player 2 wins")
