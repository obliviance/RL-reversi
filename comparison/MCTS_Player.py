from typing import List
import numpy as np
import pickle
from environment.ReversiHelpers import OthelloEnvironment

ACTION = tuple[int, int]


class MCTSPlayer:
    def __init__(self):
        #load in the mcts (this assumes the mcts file has been created)
        with open(f'.\mcts\MCTS.pickle', 'rb') as file:
            self.mcts = pickle.load(file)
        self.mcts._prev_root = None
    def pick_action(self, env: OthelloEnvironment, state: tuple[np.ndarray, int], legal_actions: List[tuple[int, int]]) -> ACTION:
        return self.mcts.mcts(env, state, legal_actions)