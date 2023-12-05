import random
import sys
import time
from comparison.HumanPlayer import HumanPlayer
from comparison.PolicyPlayer import PolicyPlayer
from comparison.RandomPlayer import RandomPlayer
from comparison.DQN_Player import DQN_Player
from environment.ReversiHelpers import DISK_BLACK, DISK_WHITE, OthelloEnvironment

def getPlayers():
    return {
        "dqn_selfplay": DQN_Player('model.dqn-selfplay'),
        "dqn": DQN_Player('model.dqn-random'),
        "random": RandomPlayer(),
        "human": HumanPlayer()
    }


if __name__ == "__main__":
    if (not (4 <= len(sys.argv) <= 5)):
        print("Invalid Input. Use \'python\' ComparePlayer.py <num_games> <player1> <player2> <optional=human|quiet>")
        quit(1)
    view = None
    verbosity = None
    if len(sys.argv) == 4:
        [_, num_games, p1_type, p2_type] = sys.argv
    else:
        [_, num_games, p1_type, p2_type, verbosity] = sys.argv
    if(verbosity == 'human'):
        view = verbosity
    num_games = int(num_games)
    players = getPlayers()
    p1 = players[p1_type]
    p2 = players[p2_type]
    if p1_type == p2_type:
        p1_type += '1'
        p2_type += '2'
    scores ={
        p1_type : 0,
        p2_type : 0,
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
        
        print("Starting round", game, "of", num_games)
        if(verbosity != 'quiet'):
            print(f"Black Player is {colors[DISK_BLACK][1]}")
            print(f"White Player is {colors[DISK_WHITE][1]}")
        else:
            round_start = time.time()
        
        terminated = False
        legal_actions = env.get_legal_moves(return_as='List')
        while not terminated:
            policy, name = colors[env.current_player]
            policy: PolicyPlayer = policy
            if(verbosity != 'quiet'):
                print(f"Player {name}'s turn!")
            if view == "human":
                env.render()
            action = policy.pick_action(env, state, legal_actions)
            state, reward, terminated, _, info = env.step(action)
            if ('error' in info):
                raise info['error']
            if ('legal_moves' in info):
                legal_actions = info['legal_moves']
            if(verbosity != 'quiet'):
                print(f"{name} plays move {action}!")
            if view == "human":
                env.render()

        winner = env.get_winner()
        if (winner in colors):
            print(f'Winner is {colors[winner][1]}!')
            scores[colors[winner][1]] += 1
        else:
            print("Game is a draw!")
            scores['draws'] += 1
        if verbosity == 'quiet':
            print('Round Finished in', time.time() - round_start,'seconds')
    print("Final Scores:")
    for name in scores:
        print(f'\t{name}: {scores[name]}')
    if verbosity == 'quiet':
        print('Games Finished in', time.time() - game_start, 'seconds')
