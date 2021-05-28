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

env = DragonWarriorEnv()
env = JoypadSpace(env, dragon_warrior_actions)
# env.reset()

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

tracemalloc.start()

# Parameters
'''Observation space (env.observation_space.shape) is 240x256x3, the height and width of the space with 3 (RBG)
color channels. The agent can take 256 different possible actions.'''
# todo add in RAM information
states = (240, 256, 3)
actions = env.action_space.n

dw_info_dict = env.state_info
dw_info_states = np.array(list(dw_info_dict.values()))

agent = DQNAgent(states=states, game_states=dw_info_states, actions=actions, max_memory=1000000,
                 double_q=True)

# Episodes
episodes = 5
rewards = []

# Timing
start = time.time()
step = 0

# Main loop
for e in range(episodes):

    # Reset env, returns screen values
    state = env.reset()
    # Return values for RAM info
    info_state = env.state_info
    dw_game_states = np.array(list(info_state.values()))

    # Reward
    total_reward = 0
    iter = 0

    # Play
    for _ in range(40001):

        # Show env (diabled), slows down learning by a factor of 3
        env.render()

        # Run agent
        action = agent.run(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)
        # actions need to be performed over 4 frames in order to be executed
        # for i in range(5):
        #     env.frame_advance(action)


        # Remember transition
        agent.add(experience=(state, next_state, action, reward, done))

        # Update agent
        agent.learn()

        # Total reward
        total_reward += reward
        print(total_reward, dragon_warrior_actions[action], iter)
        # input('press any key to advance')

        # Update state
        state = next_state

        # Increment
        iter += 1

        # If done break loop
        if done or info['exit_throne_room']:
            break

    # Rewards
    rewards.append(total_reward / iter)

    # Print
    if e % 1 == 0:
        print('Episode {e} - +'
              'Frame {f} - +'
              'Frames/sec {fs} - +'
              'Epsilon {eps} - +'
              'Mean Reward {r} - +'
              'Total Reward {R} - +'
              'Key {k} - +'
              'Gold {g} - +'
              'Torch {t}'.format(e=e,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time.time() - start)),
                                       eps=np.round(agent.eps, 4),
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
