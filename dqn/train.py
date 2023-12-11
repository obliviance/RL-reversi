import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from matplotlib import pyplot as plt
from environment.ReversiHelpers import OthelloEnvironment

import numpy as np


parser = argparse.ArgumentParser(
    prog='train',
    description='Trains a DQN')

parser.add_argument('model_name',help="Name of Model. Used as filename and outputted chart title")
parser.add_argument('-a', '--alpha', type=float, default=0.001,help="Learning Rate. Default=0.001")
parser.add_argument('-g', '--gamma', type=float, default=0.1,help="Discount Factor. Default=0.1")
parser.add_argument('-e', '--epsilon', type=float, default=0.1,help="Randomness in Policy. (Decreases to zero as you approach final episode). Default=0.1")
parser.add_argument('--episodes', type=int, default=25,help="Number of training episodes. Default=25")
parser.add_argument('--replay', type=int, default=4096,help="Size of episode replay buffer. Default=4096")
parser.add_argument('--selfplay', action='store_true',help="Train with previous iteration of agent")
parser.add_argument('--reward-shaping', action='store_true',help="Add additional per step reward based on current board")
parser.add_argument('--dqn2', action='store_true',help="Train with Neural Network with IN:(State) -> OUT:(Actions x Q(State, Actions)) [ OMIT to have IN:(State + Action) -> OUT:(Q(State,Action))]")
parser.add_argument('--evaluate', type=int, default=0,help="Number of evaluation episodes. Default=50")
if __name__ == "__main__":

    args = parser.parse_args()
    if args.dqn2:
        from dqn.dqn2 import learn_each_timestep, make_model, validate_against_random
    else:
        from dqn.dqn import learn_each_timestep, make_model, validate_against_random

    env = OthelloEnvironment()
    model = make_model([64, 64, 64], 'sigmoid', args.alpha)
    learn_each_timestep(env, model, args.episodes, gamma=args.gamma,
                        num_replay=args.replay, epsilon=args.epsilon, selfplay=args.selfplay, reward_shaping=args.reward_shaping)
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
