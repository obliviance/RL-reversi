
from typing import List
import numpy as np
from environment.ReversiHelpers import OthelloEnvironment

ACTION = tuple[int, int]


class PolicyPlayer:

    def pick_action(self, env: OthelloEnvironment, state: tuple[np.ndarray, int], legal_actions: List[tuple[int, int]]) -> ACTION:
        raise "Not Implemented"
