from environment.ReversiHelpers import DISK_BLACK, DISK_WHITE, OthelloEnvironment
from collections import deque
import random
import time
from typing import List
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np



def flatten_state_action_pair(state: tuple[np.ndarray, int], action: np.ndarray):
    return np.concatenate((
        np.array([state[1]]),
        action.flatten(),
        state[0].flatten(),
    )).flatten().astype(np.int32)


def make_model(sizes: List[int], activation: str, learning_rate: float):
    input_shape = (67, )
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for size in sizes:
        x = tf.keras.layers.Dense(size, activation=activation)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="MSE", optimizer=tf.optimizers.Adam(
        learning_rate=learning_rate))
    return model


def predict(model: tf.keras.Model, state, action):
    return model.predict([
        np.array([
            flatten_state_action_pair(
                state, action
            )
        ])
    ], verbose=0)[0]


def run_game(model: tf.keras.Model, env: OthelloEnvironment, epsilon=0.05):
    state = env.reset()
    terminated = False
    legal_actions = env.get_legal_moves(return_as="list")
    game_time = time.time()
    memory = []
    while (terminated != True):
        # Pick action that has maximum Q(s,a) for each a
        state_action_pairs = map(
            lambda action:
                flatten_state_action_pair(state, action),
                legal_actions
        )
        if (random.random() > epsilon):
            legal_actions_q_values = model.predict(
                [np.array(list(state_action_pairs))], verbose=0)
            best_action = legal_actions[np.argmax(legal_actions_q_values)]
        else:
            best_action = legal_actions[np.random.randint(
                0, len(legal_actions))]
        new_state, reward, terminated, _, info = env.step(best_action)
        if ('error' in info):
            raise info['error']
        if ('legal_moves' in info):
            legal_actions = info['legal_moves']
        memory.append((state, best_action, reward))
        state = new_state
    print("Game Timer", time.time() - game_time)
    return memory

def get_q_sa(model: tf.keras.Model, state, legal_actions: List[tuple[int, int]]):
    state_action_pairs = list(map(
        lambda action:
        flatten_state_action_pair(state, action),
        legal_actions
    ))
    legal_actions_q_values = model.predict(
        [np.array(state_action_pairs)], verbose=0)
    return legal_actions_q_values

def epsilon_greedy(model: tf.keras.Model, epsilon: float, state, legal_actions: List[tuple[int, int]]):
    if (random.random() > epsilon):
        legal_actions_q_values = get_q_sa(model, state, legal_actions)
        best_action = legal_actions[np.argmax(legal_actions_q_values)]
    else:
        best_action = legal_actions[np.random.randint(
            0, len(legal_actions))]
    return best_action

def do_step(env: OthelloEnvironment, action):
    new_state, reward, terminated, _, info = env.step(action)
    legal_actions = []
    if ('error' in info):
        raise info['error']
    if ('legal_moves' in info):
        legal_actions = info['legal_moves']

    return new_state, reward, terminated, legal_actions

def fit_sars(env, model, sars_list, gamma):
    state_actions = np.array(list(map(lambda x: flatten_state_action_pair(x[0],x[1]), sars_list)))
    q_sa_updated = []
    for sars in sars_list:
        _, _, reward, next_state = sars
        if next_state == None:
            q_sa_updated.append(0)
        else:
            player = env._action_space_to_player(next_state[1])
            legal_moves = env.get_legal_moves(
                next_state[0], return_as="list", player=player)
            if len(legal_moves) == 0:
                q_sa_updated.append(0)
            else:
                q_sa_next = np.max(get_q_sa(model, next_state, legal_moves))
                q_sa_updated.append(reward + gamma * q_sa_next)
    model.fit(state_actions, np.array(q_sa_updated))

def learn_each_timestep(env: OthelloEnvironment, model: tf.keras.Model, episodes=20, gamma=0.1, num_replay=100, epsilon=0.1,selfplay=True, pbrs=False):
    sars = deque(maxlen=num_replay)
    old_model = model
    for episode in range(episodes):

        print("Performing Game step:", episode, "Player", "Black" if env.player ==
              DISK_BLACK else "White", end=' | ')

        episode_time = time.time()

        state = env.reset()

        env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE

        terminated = False

        legal_actions = env.get_legal_moves(return_as="list")

        current_sars = [None, None, None, None]
        t=0
        while (terminated != True):

            # Each player picks action
            if env.player == env.current_player:
                action = epsilon_greedy(model, epsilon, state, legal_actions)
                current_sars[0] = state
                current_sars[1] = action

                new_state, reward, terminated, legal_actions = do_step(
                    env, action)

                current_sars[2] = reward
                if pbrs:
                    # old_state = current_sars[0]
                    # print(new_state)
                    if terminated:
                        current_sars[2] = current_sars[2] * 100
                    else:
                        current_sars[2] = np.abs(np.sum(new_state[0][new_state[0]==env.player])) 
                    # print(np.abs(np.sum(old_state[0][old_state[0]==env.player])) )

                if env.player == env.current_player:
                    current_sars[3] = new_state

            else:
                action = epsilon_greedy(
                    old_model, epsilon if selfplay else 1, state, legal_actions)
                new_state, reward, terminated, legal_actions = do_step(
                    env, action)
                if current_sars[0] is not None and env.player == env.current_player:
                    current_sars[3] = new_state

            if terminated or (not terminated and current_sars[3]) is not None:
                # For per step modelling
                # fit_sars(model, [current_sars], gamma)
                sars.append(current_sars)
                current_sars = [None,None,None,None]

            state = new_state
            t+=1
        winner = env.get_winner()
        print("Winner:", "Black" if winner ==
              DISK_BLACK else "White" if winner == DISK_WHITE else "Draw")

        if (len(sars) != 0):
            fit_sars(env, model, sars, gamma)
        print("Episode completed in \'" +
              str(time.time() - episode_time) + "\' seconds")
        old_model = model
    return

def validate_against_random(env: OthelloEnvironment, model: tf.keras.Model, episodes=50):
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

                state_action_pairs = map(
                    lambda action:
                        flatten_state_action_pair(state, action),
                        legal_actions
                )
                legal_actions_q_values = model.predict(
                    [np.array(list(state_action_pairs))], verbose=0)
                best_action = legal_actions[np.argmax(legal_actions_q_values)]
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
