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


def flatten_state(state: tuple[np.ndarray, int]):
    return np.concatenate((
        np.array([state[1]]),
        state[0].flatten(),
    )).flatten().astype(np.int32)


def make_model(sizes: List[int], activation: str, learning_rate: float):
    input_shape = (65,)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for size in sizes:
        x = tf.keras.layers.Dense(size, activation=activation)(x)
    x = tf.keras.layers.Dense(64)(x)
    outputs = tf.keras.layers.Reshape((8, 8))(x)
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
    ], verbose=0)


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
    fail_action = None
    if (random.random() > epsilon):
        p = model.predict(np.array([flatten_state(state)]), verbose=0)
        best_action = np.unravel_index(np.argmax(p[0]), p[0].shape)
        fail_action = best_action
        if (any((legal_actions[:] == best_action).all(1))):
            return best_action, "policy", None
    best_action = legal_actions[np.random.randint(
        0, len(legal_actions))]
    return best_action, "random", fail_action


def do_step(env: OthelloEnvironment, action):
    new_state, reward, terminated, truncated, info = env.step(action)
    legal_actions = []
    if ('error' in info):
        truncated = True
        reward = -10
        # raise info['error']
    if ('legal_moves' in info):
        legal_actions = info['legal_moves']

    return new_state, reward, terminated, truncated, legal_actions


def fit_sars(env, model, sars_list, gamma, verbose=0):
    states = np.array(
        list(map(lambda x: flatten_state(x[0]), sars_list)))
    next_states = np.array(
        list(map(lambda x: flatten_state(x[3]) if x[3] is not None else np.zeros((65,)), sars_list)))
    state_predictions = model.predict(states, verbose=0)
    next_state_predictions = model.predict(next_states, verbose=0)
    prediction_updated = np.zeros_like(state_predictions)
    q_sa_updated = []
    for index, sars in enumerate(sars_list):
        _, action, reward, next_state = sars
        best_next = np.max(next_state_predictions[index])
        prediction_updated[index][action[0],
                                  action[1]] = reward + gamma * best_next
    model.fit(states, prediction_updated, verbose=verbose)


def learn_each_timestep(env: OthelloEnvironment, model: tf.keras.Model, episodes=20, gamma=0.1, num_replay=100, epsilon=0.1, end_epsilon=0, selfplay=True, reward_shaping=False):
    sars = deque(maxlen=num_replay)
    old_model = model
    start_epsilon = epsilon

    for episode in range(episodes):
        epsilon = epsilon - (episode/episodes) * (start_epsilon-end_epsilon)
        print("Performing Game step:", episode, "| Player", "Black" if env.player ==
              DISK_BLACK else "White", end=' | ')

        episode_time = time.time()

        state = env.reset()

        env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE

        terminated = truncated = False

        legal_actions = env.get_legal_moves(return_as="list")

        current_sars = [None, None, None, None]
        t = 0
        action_ratios = {
            "policy": 0,
            "random": 0
        }

        prev_state = None
        prev_action = None
        reward = None
        while not (terminated or truncated):
            if env.player == env.current_player:
                best_action = legal_actions[random.randint(
                    0, len(legal_actions)-1)]
                if prev_state is not None:
                    if random.random() > epsilon:
                        pred = model.predict(np.array([flatten_state(prev_state)]),verbose=0)
                        q_vals = pred[0][legal_actions[:, 0], legal_actions[:, 1]]
                        best_action = legal_actions[np.argmax(q_vals)]
              
                prev_state = state
                prev_action = best_action
                action = best_action

                state, reward, terminated, truncated, legal_actions = do_step(
                    env, action)
                
                if reward_shaping:
                    reward += np.abs(
                        np.sum(state[0][state[0] == env.player]))/64
                
                sars.append([prev_state, prev_action, reward, state])
            else:
                if selfplay:
                    pred = old_model.predict(np.array([flatten_state(state)]),verbose=0)
                    q_vals = pred[0][legal_actions[:, 0], legal_actions[:, 1]]
                    action = legal_actions[np.argmax(q_vals)]
                else:
                    action = legal_actions[random.randint(0, len(legal_actions)-1)]
                state, reward, terminated, truncated, legal_actions = do_step(
                    env, action)
                if terminated or truncated and prev_state is not None:
                    sars.append([prev_state, prev_action, reward, state])
            t += 1

        winner = env.get_winner()
        if (terminated):
            print("Winner:", "Black" if winner ==
                  DISK_BLACK else "White" if winner == DISK_WHITE else "Draw", end=' | ')
        else:
            print("Truncated", end =' | ')
        print(f"Finished {t} episodes")
        if (len(sars) != 0):
            fit_sars(env, model, sars, gamma, verbose=2)
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
                pred = model.predict(np.array([flatten_state(state)]),verbose=0)
                q_vals = pred[0][legal_actions[:, 0], legal_actions[:, 1]]
                best_action = legal_actions[np.argmax(q_vals)]
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
