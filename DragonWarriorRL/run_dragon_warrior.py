import keras.models

from dragon_warrior_env import DragonWarriorEnv
from actions import dragon_warrior_actions
from nes_py.wrappers import JoypadSpace
from dqn_agent import DQNAgent
import time
import linecache
import os
import tracemalloc
import numpy as np
import pathlib

env = DragonWarriorEnv()
env = JoypadSpace(env, dragon_warrior_actions)
env.reset()

######
loadcheckpoint = True
path_models = pathlib.Path('models/')
filename_model = str(path_models / 'model')
episodes = 100
# render = False
render = True
pause_after_action = False
# pause_after_action = True
eps_method = 'cosine'
# eps_method = 'exp_decay'
# print_stats_per_action = True
print_stats_per_action = False
######

# Parameters
'''Observation space (env.observation_space.shape) is 240x256x3, the height and width of the space with 3 (RBG)
color channels. The agent can take 256 different possible actions.'''
# todo add in RAM information
states = (240, 256, 3)
actions = env.action_space.n

dw_info_dict = env.state_info
dw_info_states = np.array(list(dw_info_dict.values()))

agent = DQNAgent(states=states, game_states=dw_info_states, actions=actions, max_memory=1000000,
                 double_q=True, eps_method=eps_method)
if loadcheckpoint:
    agent.restore_model(filename=filename_model)
    print('Checkpoint loaded')

# Episodes
rewards = []

# Timing
start = time.time()
step = 0

def current_game_state(info_state):
    return np.array(list(info_state.values()))

# Main loop
for episode in range(episodes):

    # Reset env, returns screen values into np array
    if render == True:
        state = env.reset()
    # Return values for RAM info into np array
    game_state = current_game_state(env.state_info)

    # Reward
    total_reward = 0

    # Play
    for episode_frame in range(1001):

        if render == True:
            # Slows down learning by a factor of 3
            env.render()

        # Run agent
        action = agent.run(state=state, game_state=game_state, eps_method=eps_method)

        # Perform action
        next_state, reward, done, info = env.step(action=action)
        next_game_state = current_game_state(env.state_info)


        # Remember transition
        agent.add(experience=(state, next_state, game_state, next_game_state, action, reward, done))

        # Update agent
        agent.learn()

        # Total reward
        total_reward += reward
        if print_stats_per_action == True:
            print(np.round(total_reward, 4), dragon_warrior_actions[action],
                  episode_frame, np.round(agent.eps_now, 4))
        if pause_after_action == True:
            input('press any key to advance')

        # Update state and game_state
        state = next_state
        game_state = next_game_state

        # If done break loop
        if done or info['exit_throne_room']:
            break

    # Rewards
    rewards.append(total_reward / episode_frame)

    # todo change to average eps
    if eps_method == 'exp_decay':
        eps = agent.eps
    if eps_method == 'cosine':
        eps = agent.eps_now

    # Print
    # todo build lists/dictionaries with this info, export as csv
    if episode % 1 == 0:
        print('Episode {e} - +'
              'Frame {f} - +'
              'Frames/sec {fs} - +'
              'Epsilon {eps} - +'
              'Mean Reward {r} - +'
              'Total Reward {R} - +'
              'Key {k} - +'
              'Gold {g} - +'
              'Torch {t}'.format(e=episode,
                                 f=agent.step,
                                 fs=np.round((agent.step - step) / (time.time() - start)),
                                 eps=np.round(eps, 4),
                                 r=np.round(rewards[-1:], 4),
                                 # r=np.mean(rewards[-1:]),
                                 R=np.round(total_reward, 4),
                                 k = info['throne_room_key'],
                                 g = info['throne_room_gold'],
                                 t = info['throne_room_torch']))
        start = time.time()
        step = agent.step

# Save rewards
np.save('rewards.npy', rewards)
