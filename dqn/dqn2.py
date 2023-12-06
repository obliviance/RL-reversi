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
        # print(p)
        best_action = np.unravel_index(np.argmax(p[0]), p[0].shape)
        fail_action = best_action
        if (any((legal_actions[:] == best_action).all(1))):
            return best_action, "policy", None
        # print(best_action)
        # [q, a1, a2] = p[0]
        # legal_actions_q_values = get_q_sa(model, state, legal_actions)
        # best_action = legal_actions[np.argmax(legal_actions_q_values)]
        # best_action = (round(a1), round(a2))
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


# def fit_sars2(env: OthelloEnvironment, model, sars_list, gamma, verbose=0):
#     states = np.array(
#         list(map(lambda x: flatten_state(x[0]), sars_list)))
#     next_states = np.array(
#         list(map(lambda x: flatten_state(x[3]) if x[3] is not None else np.zeros((65,)), sars_list)))
#     state_predictions = model.predict(states, verbose=0)
#     next_state_predictions = model.predict(next_states, verbose=0)
#     prediction_updated = np.zeros_like(state_predictions)
#     q_sa_updated = []
#     for index, sars in enumerate(sars_list):
#         _, prev_action, prev_reward, state = sars
#         legal_actions = env.get_legal_moves(
#             state[0], player=env._action_space_to_player(state[1]), return_as='list')
#         q_predictions = next_state_predictions[index]
#         # print(q_predictions)
#         legal_qs = q_predictions[0][legal_actions[:,
#                                                   0], legal_actions[:, 1]]
#         max_q = np.max(legal_qs)
#         # best_action = legal_actions[np.argmax(legal_qs) ]

#         prediction_updated[index, :, :] = 0
#         prediction_updated[index, prev_action[0],
#                            prev_action[1]] = prev_reward + gamma * max_q
#     model.fit(states, prediction_updated, verbose=verbose)


def learn_each_timestep(env: OthelloEnvironment, model: tf.keras.Model, episodes=20, gamma=0.1, num_replay=100, epsilon=0.1, end_epsilon=0, selfplay=True, pbrs=False):
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
        # env.player = DISK_BLACK

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
                # if prev_state is not None:
                #     q_predictions = model.predict(
                #         np.array([flatten_state(state)]), verbose=0)
                #     # print(q_predictions)
                #     legal_qs = q_predictions[0][legal_actions[:,
                #                                               0], legal_actions[:, 1]]
                #     if random.random() > epsilon:
                #         max_q = np.max(legal_qs)
                #         best_action = legal_actions[np.argmax(legal_qs) ]
                #     else:
                #         idx = random.randint(0, len(legal_actions)-1)
                #         max_q = np.max(idx)
                #         best_action = legal_actions[idx]

                #     q_predictions[0, :, :] = 0
                #     q_predictions[0, best_action[0],
                #                   best_action[1]] = prev_reward + gamma * max_q
                #     model.fit(np.array([flatten_state(state)]), q_predictions)
                prev_state = state
                prev_action = best_action
                action = best_action

                state, reward, terminated, truncated, legal_actions = do_step(
                    env, action)
                
                if pbrs:
                    # old_state = current_sars[0]
                    # print(new_state)
                    reward = np.abs(
                        np.sum(state[0][state[0] == env.player]))/64
                
                sars.append([prev_state, prev_action, reward, state])
                # if terminated or truncated:
                #     model.fit(
                #         np.array([flatten_state(state)]), np.zeros((1, 8, 8)))
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
                #     model.fit(
                #         np.array([flatten_state(prev_state)]), np.zeros((1, 8, 8)))

                # if random.random() > epsilon:
                # for action in legal_actions:

                # action_update = []
                # action_value =
                # action_update.append(action)
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
            # fit_sars2(env, model, sars, gamma, verbose=2)
        print("Episode completed in \'" +
              str(time.time() - episode_time) + "\' seconds")
        old_model = model
    return


# def learn_each_timestep(env: OthelloEnvironment, model: tf.keras.Model, episodes=20, gamma=0.1, num_replay=100, epsilon=0.1, selfplay=True, pbrs=False):
#     sars = deque(maxlen=num_replay)
#     old_model = model
#     for episode in range(episodes):

#         print("Performing Game step:", episode, "Player", "Black" if env.player ==
#               DISK_BLACK else "White", end=' | ')

#         episode_time = time.time()

#         state = env.reset()

#         env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE

#         terminated = truncated = False

#         legal_actions = env.get_legal_moves(return_as="list")

#         current_sars = [None, None, None, None]
#         t = 0
#         action_ratios = {
#             "policy": 0,
#             "random": 0
#         }

#         prev_state = None

#         while not (terminated or truncated):

#             # Each player picks action
#             if env.player == env.current_player:

#                 action, action_type, fail_action = epsilon_greedy(
#                     model, epsilon, state, legal_actions)

#                 action_ratios[action_type] += 1
#                 current_sars[0] = state
#                 current_sars[1] = action
#                 if (fail_action):
#                     temp_sars = current_sars.copy()
#                     temp_sars[2] = -100
#                     sars.append(temp_sars)

#                 new_state, reward, terminated, truncated, legal_actions = do_step(
#                     env, action)

#                 current_sars[2] = reward
#                 if pbrs:
#                     # old_state = current_sars[0]
#                     # print(new_state)
#                     if terminated:
#                         current_sars[2] = current_sars[2] * 100
#                     else:
#                         current_sars[2] = np.abs(
#                             np.sum(new_state[0][new_state[0] == env.player]))
#                     # print(np.abs(np.sum(old_state[0][old_state[0]==env.player])) )

#                 if env.player == env.current_player:
#                     current_sars[3] = new_state

#             else:
#                 action, _, _ = epsilon_greedy(
#                     old_model, epsilon if selfplay else 1, state, legal_actions)
#                 new_state, reward, terminated, truncated, legal_actions = do_step(
#                     env, action)
#                 if current_sars[0] is not None and env.player == env.current_player:
#                     current_sars[3] = new_state

#             if (terminated or truncated) or (not terminated and not truncated and current_sars[3] is not None):
#                 # For per step modelling
#                 # fit_sars(env, model, [current_sars], gamma, verbose=0)
#                 sars.append(current_sars)
#                 current_sars = [None, None, None, None]

#             state = new_state
#             t += 1
#         print(
#             f"Finished {t} episodes | Actions = {action_ratios['policy']} | Action Ratio = {round(action_ratios['policy']/(action_ratios['policy'] + action_ratios['random']),2)} | Epsilon = {epsilon} | ", end='')
#         winner = env.get_winner()
#         if (terminated):
#             print("Winner:", "Black" if winner ==
#                   DISK_BLACK else "White" if winner == DISK_WHITE else "Draw")
#         else:
#             print("Truncated")
#         if (len(sars) != 0):
#             fit_sars(env, model, sars, gamma, verbose=2)
#         print("Episode completed in \'" +
#               str(time.time() - episode_time) + "\' seconds")
#         old_model = model
#     return


# def validate_against_random(env: OthelloEnvironment, model: tf.keras.Model, episodes=50):
#     scores = []
#     for e in range(episodes):
#         state = env.reset()
#         env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE
#         print("Game", e, "Player", "Black" if env.player ==
#               DISK_BLACK else "White", end=' ')
#         terminated = False
#         legal_actions = env.get_legal_moves(return_as="list")
#         game_time = time.time()
#         while (terminated != True):
#             if env.current_player == env.player:

#                 state_action_pairs = map(
#                     lambda action:
#                         flatten_state_action_pair(state, action),
#                         legal_actions
#                 )
#                 legal_actions_q_values = model.predict(
#                     [np.array(list(state_action_pairs))], verbose=0)
#                 best_action = legal_actions[np.argmax(legal_actions_q_values)]
#             else:
#                 best_action = legal_actions[np.random.choice(
#                     len(legal_actions), 1)][0]
#             new_state, reward, terminated, _, info = env.step(best_action)
#             if ('error' in info):
#                 raise info['error']
#             if ('legal_moves' in info):
#                 legal_actions = info['legal_moves']
#             if (terminated):
#                 scores.append(reward)
#                 print("Score", reward)
#             state = new_state
#         print("Game Timer", time.time() - game_time)

#         print("Done!")
#     return scores
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
