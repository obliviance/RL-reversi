
from environment.ReversiHelpers import DISK_BLACK, DISK_WHITE, OthelloEnvironment
import random
import time
import math
import copy
import numpy as np

class TerminalStateException(Exception): pass

class Node:
    def __init__(self, state, valid_moves):
        self._board, self._player = state

        self._visits = 0
        self._children = []
        self._child_values = None
        self._child_visits = None

        self._state = state
        self._valid_moves = valid_moves
        self._valid_moves_indices = None
        self._p_v = None

        self._simulated = 0
        self._wins = 0

        self._parent_action = None
        self._opponent_action = None
        self._parent = None

    @property
    def board(self):
        return self._board
    
    @property
    def player(self):
        return self._player

    @property
    def visits(self):
        return self._visits
    
    @property
    def children(self):
        return self._children

    @property
    def child_values(self):
        return self._child_values

    @property
    def child_visits(self):
        return self._child_visits
    
    @property
    def state(self):
        return self._state

    @property
    def valid_positions(self):
        return self._valid_moves

    def set_children(self, children):
        self._children = children
        self._child_values = np.zeros(len(children), dtype=np.float32)
        self._child_visits = np.zeros(len(children), dtype=np.int32)

    def increment_visits(self):
        self._visits += 1

    def update_value(self, child_idx, value):
        self._child_visits[child_idx] += 1
        self._child_values[child_idx] += value
    
    def update(self, reward):
        self._simulated += 1
        if (reward == 1):
            self._wins += 1
    def set_parent(self, node):
        self._parent = node
    def set_parent_action(self, action):
        self._parent_action = action
    def set_opponent_action(self, action):
        self._opponent_action = action


class MCTS:
    def __init__(self, n_iter=50, c=2.):
        self._n_iter = int(n_iter)
        self._c = float(c)
        self._n_simulated = 0
        self._nodes = {}
        self._prev_root = None
    
    def mcts(self, env, state, legal_actions):
        length = len(legal_actions)
        if (length == 1) | (length == 2):
            return legal_actions[0]
        my_env = copy.deepcopy(env) #make a copy so we don't mess up the original during traversal.
        if (self._prev_root == None):
            s = state
            root = Node(s, legal_actions)
        else:
            prev = self._prev_root
            root = None
            for child in prev._children:
                if (child._board==state[0]).all():
                    root = child
                    break
            if root == None:
                #we couldn't find the nex root as a child of the previous node.
                s = state
                root = Node(s, legal_actions)

        if len(root.valid_positions) == 0:
            raise TerminalStateException()
        #initial traversal
        self.traverse(root, self._c, my_env, init = True)
        for i in range(self._n_iter):
            self.traverse(root, self._c, my_env)

        #pick the child with the best win rate
        max = 0
        max_child = root._children[0]
        for child in root._children:
            if (child._simulated > 0):
                if (child._wins/child._simulated) > max:
                    max = (child._wins/child._simulated)
                    max_child = child
        #choose the action that leads to this child
        self._prev_root = root
        return max_child._parent_action

    def traverse(self, node, c, env, init = False):
        if init:
            #maybe visited this node in a past iteration.
            if len(node._children) == 0:
                expand_node(node, env)
            return
        else:
            #is the current node a leaf?
            if len(node._children) == 0:
                if len(node.valid_positions) == 0:
                    #this should be an error
                    raise TerminalStateException()
                if (node._simulated == 0):
                    #Rollout
                    rollout(node, env)
                    self._n_simulated += 1
                else:
                    #expand:
                    expand_node(node, env)
                    if len(node._children) == 0:
                        #this means we expanded but still have no children since we won.
                        return
                    #every child is completely new, rollout the first.
                    child = node._children[0]
                    #need to step the enviroment to their start
                    new_env = copy.deepcopy(env)
                    new_env.step(child._parent_action)
                    new_env.step(child._opponent_action)
                    rollout(child, new_env)
                    self._n_simulated += 1
            else:
                #traverse env and tree recursively until a leaf is found.
                #pick child with highest UCT score
                scores = self.uct_score(node, c)
                idx = np.argmax(scores)
                child = node._children[idx]
                new_env = copy.deepcopy(env)
                new_env.step(child._parent_action)
                new_env.step(child._opponent_action)
                while len(child._children) != 0:
                    #pick child with highest UCT score
                    idx = np.argmax(self.uct_score(child, c))
                    child = child._children[idx]
                    new_env.step(child._parent_action)
                    new_env.step(child._opponent_action)
                #good, now traverse.
                self.traverse(child, self._c, new_env)

    def uct_score(self, node, c):
        #calculates utc scores for each child node.
        children = node._children
        scores = []
        for child in children:
            if child._simulated == 0:
                #very large number
                scores.append(1e3)
            else:
                scores.append((child._wins/child._simulated) + c*math.sqrt(math.log(self._n_simulated)/child._simulated))
        return scores


def expand_node(node, env):
    #just expand children.
    legal_actions = node._valid_moves
    children = []
    for action in legal_actions:
        #Expand
        copy_env = copy.deepcopy(env) #copy the enviroment
        #Do the my action
        state, reward, terminated, _, info = copy_env.step(action)
        if terminated:
            #Backpropagate
            backpropagate(node, reward)
        elif node._player == state[1]:
            #this means we are at another decision point, so save this as a child:
            parent_action = action
            #finally create and add this child.
            legal_actions2 = info['legal_moves']
            child = Node(state, legal_actions2)
            child.set_parent(node)
            child.set_parent_action(parent_action)
            children.append(child)
        else:
            parent_action = action
            #expand the opponents moves
            legal_actions2 = info['legal_moves']
            for action2 in legal_actions2:
                copy_env2 = copy.deepcopy(copy_env) #copy the enviroment
                #Do the opponents action
                state, reward, terminated, _, info = copy_env2.step(action2)
                if terminated:
                    #Backpropagate
                    backpropagate(node, reward)
                else:
                    opponent_action = action2
                    #finally create and add this child.
                    legal_actions3 = info['legal_moves']
                    child = Node(state, legal_actions3)
                    child.set_parent(node)
                    child.set_parent_action(parent_action)
                    child.set_opponent_action(opponent_action)
                    children.append(child)
    node.set_children(children)

def backpropagate(node, reward):
    node.update(reward)
    parent = node._parent
    while parent != None:
        parent.update(reward)
        parent = parent._parent

def rollout(node, env):
    #copy the enviroment so we don't change it.
    copy_env = copy.deepcopy(env)
    #play until the end.
    state = node._state
    terminated = False
    legal_actions = node._valid_moves
    while (terminated != True):
        #pick a random choice
        best_action = legal_actions[np.random.choice(
            len(legal_actions), 1)][0]
        new_state, reward, terminated, _, info = copy_env.step(best_action)
        if ('error' in info):
            raise info['error']
        if ('legal_moves' in info):
            legal_actions = info['legal_moves']
        if (terminated):
            backpropagate(node, reward)
        state = new_state


#def learn_each_timestep(env, episodes, gamma, epsilon):
#    tree_search = MCTS()
#    for e in episodes:
#        state = env.reset()
#        env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE
#        print("Game", e, "Player", "Black" if env.player ==
#              DISK_BLACK else "White", end=' ')
#        terminated = False
#        legal_actions = env.get_legal_moves(return_as="list")
#        game_time = time.time()
#        while (terminated != True):
#            if env.current_player == env.player:
#                #select best action
#                board, player = state
#                best_action = tree_search.mcts(env, state, legal_actions)
#            else:
#                best_action = legal_actions[np.random.choice(
#                    len(legal_actions), 1)][0]
#            new_state, reward, terminated, _, info = env.step(best_action)
#            if ('error' in info):
#                raise info['error']
#            if ('legal_moves' in info):
#                legal_actions = info['legal_moves']
#            if (terminated):
#                print("Score", reward)
#                #backpropagate???
#            state = new_state
        


def validate_against_random(tree_search, env, episodes=50):
    scores = []
    for e in range(episodes):
        state = env.reset()
        env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE
        print("Game", e, "Player", "Black" if env.player ==
              DISK_BLACK else "White", end=' ')
        terminated = False
        legal_actions = env.get_legal_moves(return_as="list")
        game_time = time.time()
        while (terminated != True):
            if env.current_player == env.player:
                #select best action
                board, player = state
                best_action = tree_search.mcts(env, state, legal_actions)
            else:
                best_action = legal_actions[np.random.choice(
                    len(legal_actions), 1)][0]
            new_state, reward, terminated, _, info = env.step(best_action)
            if ('error' in info):
                raise info['error']
            if ('legal_moves' in info):
                legal_actions = info['legal_moves']
            if (terminated):
                scores.append(reward)
                print("Score", reward)
            state = new_state
        print("Game Timer", time.time() - game_time)

        print("Done!")
    return scores