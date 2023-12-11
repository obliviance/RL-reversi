import argparse
import pickle
from matplotlib import pyplot as plt
from environment.ReversiHelpers import OthelloEnvironment
from mcts.mcts import validate_against_random, MCTS

import numpy as np


parser = argparse.ArgumentParser(
    prog='train',
    description='Trains a MonteCarlo')

parser.add_argument('model_name')
parser.add_argument('--evaluate', type=int, default=0)
if __name__ == "__main__":
    
    args = parser.parse_args()

    env = OthelloEnvironment()

    if (True):
        tree_search = MCTS()
        scores = validate_against_random(tree_search, env, args.evaluate)

        scores = np.array(scores)
        cumulative_loss = np.cumsum([1 if x < 0 else 0 for x in scores])
        cumulative_wins = np.cumsum([1 if x > 0.9 else 0 for x in scores])
        cumulative_draws = np.cumsum(
            [1 if np.abs(x-0.5) <= 0.05 else 0 for x in scores])

        fig, axes = plt.subplots(1, 1)
        print('Scores', scores)
        print('Wins', cumulative_wins[-1])
        print('Draws', cumulative_draws[-1])
        print('Losses', cumulative_loss[-1])
        axes.set_title(
            f"MonteCarloTreeSearch ( - Wins over time vs Random Player)")
        axes.plot(cumulative_wins, label='Wins')
        axes.plot(cumulative_draws, label='Draws')
        axes.plot(cumulative_loss, label='Loss')
        axes.legend()
        fig.savefig(f'MonteCarloTreeSearch.png')
        with open(f'\mcts\MCTS.pickle', 'wb') as file:
            pickle.dump(tree_search, file) 
