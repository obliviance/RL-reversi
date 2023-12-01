# %%
from collections import deque
from itertools import combinations_with_replacement
import random
import time
from typing import List
import pygame
import tensorflow as tf
import matplotlib.pyplot as plt
# %%
import numpy as np

from environment.ReversiHelpers import DISK_BLACK, DISK_WHITE, OthelloEnvironment


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
        turn_time = time.time()
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
        # print("Turn Timer", time.time() - turn_time)
    print("Game Timer", time.time() - game_time)
    return memory


def get_per_step_labels(winner, memory, gamma):
    xs = []
    ys = []
    g = 1
    for index, (state, action, reward) in enumerate(memory):
        current_player = state[1]
        if (index == 0):
            long_term_reward = reward
        else:
            long_term_reward = reward + g * \
                (1 if current_player == winner else -1)
        g *= gamma
        xs.append(flatten_state_action_pair(state, action))
        ys.append(long_term_reward)
    return xs, ys


def get_per_step_labels2(winner, memory, gamma):
    xs = []
    ys = []
    # g = 1
    for index, (state, action, reward) in enumerate(memory):
        current_player = state[1]
        if (index == 0):
            long_term_reward = reward
        else:
            long_term_reward = reward + (gamma * ys[index-1])
        # g *= gamma
        xs.append(flatten_state_action_pair(state, action))
        ys.append(long_term_reward)
    return xs, ys


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

# DEFUNCT - Past Bad implementation of DQN
def learn_each_timestep(env: OthelloEnvironment, model: tf.keras.Model, episodes=20, gamma=0.9, step_size=0.05, num_replay=100, epsilon=0.05):
    memory_xs = deque(maxlen=num_replay)
    memory_ys = deque(maxlen=num_replay)
    sars = deque(maxlen=num_replay)
    old_model = model
    for episode in range(episodes):
        print("Performing Game step:", episode, "Player", "Black" if env.player ==
              DISK_BLACK else "White", end=' | ')
        episode_time = time.time()
        state = env.reset()

        prev_state = None
        prev_action = None

        env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE

        terminated = False

        legal_actions = env.get_legal_moves(return_as="list")

        if env.player == env.current_player:
            action = epsilon_greedy(model, epsilon, state, legal_actions)
        else:
            action = epsilon_greedy(old_model, epsilon, state, legal_actions)

        while (terminated != True):

            if env.player == env.current_player:
                prev_state = state
                prev_action = action

            # Perform turn
            new_state, reward, terminated, legal_actions = do_step(env, action)

            if env.player == env.current_player and (prev_state is None or prev_action is None):
                next_action = epsilon_greedy(
                    model, epsilon, state, legal_actions)
            # If my turn, time to fit
            elif env.player == env.current_player:

                Q_sa = predict(model, prev_state, prev_action)
                if terminated:
                    Q_sa_next = 0
                else:
                    # print(prev_state, prev_action)
                    prev_state_action = flatten_state_action_pair(
                        prev_state,
                        prev_action)

                    # next_state_action = flatten_state_action_pair(
                    #     new_state,
                    #     next_action)

                    Q_sa_next = np.max(get_q_sa(model, state, legal_actions))

                    next_action = epsilon_greedy(
                        model, epsilon, state, legal_actions)

                Q_sa_updated = Q_sa + step_size * \
                    (reward + gamma * Q_sa_next - Q_sa)
                # print(Q_sa, Q_sa_updated, Q_sa_updated-Q_sa)
                model.fit(np.array([prev_state_action]),
                          np.array([Q_sa_updated]), verbose=0)
                memory_xs.append(prev_state_action)
                memory_ys.append(reward)

            else:
                if terminated:
                    Q_sa = predict(model, prev_state, prev_action)
                    Q_sa_next = 0
                    Q_sa_updated = Q_sa + step_size * \
                        (reward + gamma * Q_sa_next - Q_sa)
                    model.fit(np.array([prev_state_action]),
                              np.array([Q_sa_updated]), verbose=0)
                    memory_xs.append(prev_state_action)
                    memory_ys.append(reward)

                else:
                    next_action = epsilon_greedy(
                        old_model, epsilon, state, legal_actions)

            state = new_state
            action = next_action

        winner = env.get_winner()
        print("Winner:", "Black" if winner ==
              DISK_BLACK else "White" if winner == DISK_WHITE else "Draw")

        if (len(memory_xs) != 0):
            # print(memory)

            model.fit(
                np.array(memory_xs),
                np.array(memory_ys))
        print("Episode completed in \'" +
              str(time.time() - episode_time) + "\' seconds")
        old_model = model
    return


def fit_sars(model, sars_list, gamma):
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
    # if len(sars_list) > 1:
    #     pred = model.pred(np.array(state_actions)).flatten()
    #     print(list(zip(pred,q_sa_updated)))
    model.fit(state_actions, np.array(q_sa_updated))


def learn_each_timestep2(env: OthelloEnvironment, model: tf.keras.Model, episodes=20, gamma=0.9, step_size=0.05, num_replay=100, epsilon=0.05):
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
        while (terminated != True):

            # Each player picks action
            if env.player == env.current_player:
                action = epsilon_greedy(model, epsilon, state, legal_actions)
                current_sars[0] = state
                current_sars[1] = action

                new_state, reward, terminated, legal_actions = do_step(
                    env, action)

                current_sars[2] = reward

                if env.player == env.current_player:
                    current_sars[3] = new_state

            else:
                action = epsilon_greedy(
                    old_model, epsilon, state, legal_actions)
                new_state, reward, terminated, legal_actions = do_step(
                    env, action)
                if current_sars[0] is not None and env.player == env.current_player:
                    current_sars[3] = new_state

            if terminated or (not terminated and current_sars[3]) is not None:
                # For per step modelling
                fit_sars(model, [current_sars], gamma)
                sars.append(current_sars)
                current_sars = [None,None,None,None]

            state = new_state
        winner = env.get_winner()
        print("Winner:", "Black" if winner ==
              DISK_BLACK else "White" if winner == DISK_WHITE else "Draw")

        if (len(sars) != 0):
            fit_sars(model, sars, gamma)
        print("Episode completed in \'" +
              str(time.time() - episode_time) + "\' seconds")
        old_model = model
    return

# DEFUNCT - Past SARSA method
def learn_each_episode(env: OthelloEnvironment, model: tf.keras.Model, episodes=20, gamma=0.9, step_size=0.05, num_replay=100, epsilon=0.05):
    memory_xs = deque(maxlen=num_replay)
    memory_ys = deque(maxlen=num_replay)
    sars = deque(maxlen=num_replay)
    old_model = model
    for episode in range(episodes):
        print("Performing Game step:", episode, "Player", "Black" if env.player ==
              DISK_BLACK else "White", end=' | ')
        episode_time = time.time()
        state = env.reset()

        episode_states = []
        episode_possible_actions = []
        episode_policy_actions = []
        episode_rewards = []

        env.player = DISK_BLACK if random.randint(0, 1) == 1 else DISK_WHITE

        terminated = False

        legal_actions = env.get_legal_moves(return_as="list")

        while (terminated != True):
            if env.player == env.current_player:
                action = epsilon_greedy(model, epsilon, state, legal_actions)
                episode_states.append(state)
                episode_policy_actions.append(action)
                episode_possible_actions.append(legal_actions)

                # Perform turn
                new_state, reward, terminated, legal_actions = do_step(
                    env, action)

                episode_rewards.append(reward)

                # if (terminated or env.player == env.current_player):
                #     memory_xs.append(flatten_state_action_pair(
                #         episode_states[-1], episode_policy_actions[-1]))
                #     memory_ys.append(reward)
            else:
                action = epsilon_greedy(
                    old_model, epsilon, state, legal_actions)
                # Perform turn
                new_state, reward, terminated, legal_actions = do_step(
                    env, action)

            state = new_state

        xs = list(map(lambda x: flatten_state_action_pair(
            x[0], x[1]), zip(episode_states, episode_policy_actions)))

        # possible_state_actions = [flatten_state_action_pair(s, a) for a in (x[1] for x in zip(episode_states,possible_state_actions))]
        possible_state_actions = []
        for s, action_list in zip(episode_states, episode_possible_actions):
            for a in action_list:
                possible_state_actions.append(flatten_state_action_pair(s, a))

        q_sa = model.predict(np.array(xs))
        q_sa_all_next = model.predict(np.array(possible_state_actions))
        q_sa_next = []
        current_index = 0
        for index, action_list in enumerate(episode_possible_actions):
            if (index == len(episode_possible_actions) - 1):
                q_sa_next.append(0)
                continue
            q_sa_next.append(
                np.max(q_sa_all_next[current_index:current_index+len(action_list)]))
            current_index += len(action_list)
        # q_sa_next = np.array(q_sa_next)
        q_sa_updated = np.array([
            episode_rewards[index] + gamma * max_q_sa_next
            for index, max_q_sa_next in enumerate(q_sa_next[1:])
        ] + [0])
        # for index, max_q_sa_next in enumerate(q_sa_next):
        #     q_sa_updated = episode_rewards[index] + \
        #         gamma*max_q_sa_next-q_sa[index]
        # print(xs)
        # print(np.array(xs).shape)
        # print(q_sa_updated)
        # print(q_sa_updated.shape)
        model.fit(np.array(xs), [q_sa_updated])

        winner = env.get_winner()
        print("Winner:", "Black" if winner ==
              DISK_BLACK else "White" if winner == DISK_WHITE else "Draw")

        # if (len(memory_xs) != 0):
        #     # print(memory)

        #     model.fit(
        #         np.array(memory_xs),
        #         np.array(memory_ys))
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
            # turn_time = time.time()
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
                # print(best_action)
            new_state, reward, terminated, _, info = env.step(best_action)
            if ('error' in info):
                raise info['error']
            if ('legal_moves' in info):
                legal_actions = info['legal_moves']
            if (terminated):
                scores.append(reward)
                print("Score", reward)
            state = new_state
            # print("Turn Timer", time.time() - turn_time)
        print("Game Timer", time.time() - game_time)

        print("Done!")
    return scores
# %%


env = OthelloEnvironment()
model = make_model([64], 'sigmoid', 0.001)
learn_each_timestep2(env, model, 50, gamma=0.05,
                   num_replay=2048, step_size=0.05, epsilon=0.1)
# %%
scores = validate_against_random(env, model, 50)
# %%
scores = np.array(scores)
cumulative_loss = np.cumsum([1 if x < 0 else 0 for x in scores])
cumulative_wins = np.cumsum([1 if x > 0.9 else 0 for x in scores])
cumulative_draws = np.cumsum(
    [1 if np.abs(x-0.5) <= 0.05 else 0 for x in scores])

fig, axes = plt.subplots(1, 1)
print(scores)
axes.plot(cumulative_wins, label='wins')
axes.plot(cumulative_draws, label='draws')
axes.plot(cumulative_loss, label='loss')
axes.legend()

# %%
