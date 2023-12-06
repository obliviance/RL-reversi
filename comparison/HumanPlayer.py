import random
from typing import List
import numpy as np
from comparison.PolicyPlayer import ACTION, PolicyPlayer
from environment.ReversiHelpers import OthelloEnvironment

class HumanPlayer(PolicyPlayer):
    
    def pick_action(self, env : OthelloEnvironment, state: tuple[np.ndarray, int], legal_actions:np.ndarray) -> ACTION:
        action = None
        while True:
            
            input_string = input("Please choose a position to place your piece, in the form 'row col': ")
            input_values = input_string.split(' ')
            # Verify if all 2 inputs are digits
            if(not(len(input_values) == 2 and all([element.isdigit() for element in input_values]))):
                print("Invalid Input")
                continue
            action = tuple(map(int,input_values))
            if(any((legal_actions[:] == action).all(1))):
                break
            else:
                print("Invalid Input!")
        return action