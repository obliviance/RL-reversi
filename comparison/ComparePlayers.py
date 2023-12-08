import argparse
import random
import sys
import time
from comparison.HumanPlayer import HumanPlayer
from comparison.PolicyPlayer import PolicyPlayer
from comparison.RandomPlayer import RandomPlayer
from comparison.DQN_Player import DQN2_Player, DQN_Player
from environment.ReversiHelpers import DISK_BLACK, DISK_WHITE, OthelloEnvironment


def getPlayers():
    try:
        return {
            "dqn": DQN_Player('model.dqn'),
            "dqn-rs": DQN_Player('model.dqn-rs'),
            "dqn-selfplay": DQN_Player('model.dqn-selfplay'),
            "dqn-rs-selfplay": DQN_Player('model.dqn-rs-selfplay'),
            "dqn2": DQN2_Player('model.dqn2'),
            "dqn2-rs": DQN2_Player('model.dqn2-rs'),
            "dqn2-selfplay": DQN2_Player('model.dqn2-selfplay'),
            "dqn2-rs-selfplay": DQN2_Player('model.dqn2-rs-selfplay'),
            "random": RandomPlayer(),
            "human": HumanPlayer()
        }
    except OSError:
        raise "Did you export the models from models.zip into the root directory?"


parser = argparse.ArgumentParser(
    prog='ComparePlayers',
    description='Compare players against one another')

parser.add_argument('p1')
parser.add_argument('p2')
parser.add_argument('--games', type=int, default=1)
parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2])
parser.add_argument('--human-render', action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    p1_type = args.p1
    p2_type = args.p2

    view = "human" if args.human_render else None
    verbosity = args.verbosity

    num_games = args.games

    players = getPlayers()
    p1 = players[p1_type]
    p2 = players[p2_type]
    if p1_type == p2_type:
        p1_type += '1'
        p2_type += '2'
    scores = {
        p1_type: 0,
        p2_type: 0,
        'draws': 0
    }
    game_start = time.time()
    for game in range(num_games):
        
        env = OthelloEnvironment(
            render_mode=view if view is not None else None)
        state = env.reset()
        if random.random() > 0.5:
            colors = {
                DISK_BLACK: [p1, p1_type],
                DISK_WHITE: [p2, p2_type]
            }
        else:
            colors = {
                DISK_BLACK: [p2, p2_type],
                DISK_WHITE: [p1, p1_type]
            }
        if verbosity > 0:
            print("Starting round", game, "of", num_games)
        if verbosity > 1:
            print(f"Black Player is {colors[DISK_BLACK][1]}")
            print(f"White Player is {colors[DISK_WHITE][1]}")
        else:
            round_start = time.time()

        terminated = False
        legal_actions = env.get_legal_moves(return_as='List')
        while not terminated:
            policy, name = colors[env.current_player]
            policy: PolicyPlayer = policy
            if verbosity > 1:
                print(f"Player {name}'s turn!")
            if view:
                env.render()
            action = policy.pick_action(env, state, legal_actions)
            state, reward, terminated, _, info = env.step(action)
            if ('error' in info):
                raise info['error']
            if ('legal_moves' in info):
                legal_actions = info['legal_moves']
            if verbosity > 1:
                print(f"{name} plays move {action}!")
            if view:
                env.render()

        winner = env.get_winner()
        if (winner in colors):
            if verbosity > 0:
                print(f'Winner is {colors[winner][1]}!')
            scores[colors[winner][1]] += 1
        else:
            if verbosity > 0:
                print("Game is a draw!")
            scores['draws'] += 1
        if verbosity > 0:
            print('Round Finished in', time.time() - round_start, 'seconds')
    print("Final Scores:")
    for name in scores:
        print(f'\t{name}: {scores[name]}')
    print('Games Finished in', time.time() - game_start, 'seconds')
