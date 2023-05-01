class Policy:
    def __call__(self, s: int, a: int) -> float:
        pass


class RandomPolicyTicTacToe(Policy):
    def __call__(self, s: int, a: int) -> float:
        if a == 1:
            return 1.0
        return 0.0
