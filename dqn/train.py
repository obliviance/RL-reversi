
import argparse
from matplotlib import pyplot as plt

import numpy as np


parser = argparse.ArgumentParser(
    prog='train',
    description='Trains a DQN')

parser.add_argument('model_name')
parser.add_argument('-a', '--alpha', type=float, default=0.001)
parser.add_argument('-g', '--gamma', type=float, default=0.1)
parser.add_argument('-e', '--epsilon', type=float, default=0.1)
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--replay', type=int, default=2048)
parser.add_argument('--selfplay', action='store_true')
parser.add_argument('--pbrs', action='store_true')
parser.add_argument('--evaluate', type=int, default=0)
if __name__ == "__main__":
    from dqn.dqn import learn_each_timestep, make_model, validate_against_random
    from environment.ReversiHelpers import OthelloEnvironment

    args = parser.parse_args()
    env = OthelloEnvironment()
    model = make_model([64], 'sigmoid', args.alpha)
    learn_each_timestep(env, model, args.episodes, gamma=args.gamma,
                        num_replay=args.replay, epsilon=args.epsilon, selfplay=args.selfplay, pbrs=args.pbrs)
    model.save(args.model_name)

    if (args.evaluate > 0):
        scores = validate_against_random(env, model, args.evaluate)

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
            f"DQN ({args.model_name} - Wins over time vs Random Player)")
        axes.plot(cumulative_wins, label='Wins')
        axes.plot(cumulative_draws, label='Draws')
        axes.plot(cumulative_loss, label='Loss')
        axes.legend()
        fig.savefig(f'DQN-{args.model_name}.png')

        model.save('model.dqn')
