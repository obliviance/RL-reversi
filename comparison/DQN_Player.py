from typing import List
import numpy as np
from comparison.PolicyPlayer import PolicyPlayer, ACTION
from environment.ReversiHelpers import OthelloEnvironment


def flatten_state_action_pair(state: tuple[np.ndarray, int], action: np.ndarray):
    return np.concatenate((
        np.array([state[1]]),
        action.flatten(),
        state[0].flatten(),
    )).flatten().astype(np.int32)


class DQN_Player(PolicyPlayer):
    def __init__(self, model_name:str):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_name)
        
    def pick_action(self, env : OthelloEnvironment, state: tuple[np.ndarray, int], legal_actions:List[tuple[int, int]]) -> ACTION:
        state_action_pairs = list(map(lambda x: flatten_state_action_pair(state, x), legal_actions))
        q_values = self.model.predict(np.array(state_action_pairs),verbose=0)
        best_action = legal_actions[np.argmax(q_values)]
        return best_action   