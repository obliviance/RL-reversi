import random
from typing import List
import numpy as np
from comparison.PolicyPlayer import ACTION, PolicyPlayer
from environment.ReversiHelpers import OthelloEnvironment


class RandomPlayer(PolicyPlayer):
    
    def pick_action(self, env : OthelloEnvironment, state: tuple[np.ndarray, int], legal_actions:List[tuple[int, int]]) -> ACTION:
        return random.choice(legal_actions)