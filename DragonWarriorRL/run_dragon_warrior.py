import pathlib
import time
import datetime
import numpy as np
import pandas as pd
from nes_py.wrappers import JoypadSpace
from environmentwrappers import ButtonRemapper

# from dumb_dw_env import DumbDragonWarriorEnv
from actions import dragon_warrior_actions, dragon_warrior_comboactions
from dqn_agent import DQNAgent
from dragon_warrior_env import DragonWarriorEnv

######
episodes = 20
frames_per_episode = 20000
loadcheckpoint = True
# loadcheckpoint = False
# renderflag = False
renderflag = True
print_stats_per_action = True
# print_stats_per_action = False
pause_after_action = False
# pause_after_action = True

use_dumb_dw_env = False
# todo adjust cosine method, or change from boolean to variable controlling period of cosine
eps_method = 'cosine'
# eps_method = 'exp_decay'
eps_cosine_method_frames_per_cycle = 500  # travels one wavelength in this
frames_to_elapse_before_saving_agent = 20000
path_models = pathlib.Path('models/')
filename_model = str(path_models / 'model')
######

if use_dumb_dw_env == True:
    env = DumbDragonWarriorEnv()
else:
    env = DragonWarriorEnv()
# env = JoypadSpace(env, dragon_warrior_actions)
env = ButtonRemapper(env, dragon_warrior_actions, dragon_warrior_comboactions, renderflag=renderflag)# actions=dragon_warrior_comboactions, renderflag=renderflag)
env.reset()

# Parameters
'''Observation space (env.observation_space.shape) is 240x256x3, the height and width of the space with 3 (RBG)
color channels. The agent can take 256 different possible actions.'''
# todo add in RAM information
states = (240, 256, 3)
actions = env.action_space.n
action_space_combo = env.action_space_combo.n

dw_info_dict = env.state_info
dw_info_states = np.array(list(dw_info_dict.values()))

agent = DQNAgent(states=states, game_states=dw_info_states, actions=action_space_combo, max_memory=1000000,
                 double_q=True, eps_method=eps_method, agent_save=frames_to_elapse_before_saving_agent)
if loadcheckpoint:
    agent.restore_model(filename=filename_model)
    print('Checkpoint loaded')

# Episodes
rewards = []
episode_number = []
frames_in_episode = []
epsilon_at_end = []
key_found_in_episode = []
gold_found_in_episode = []
torch_found_in_episode = []

# Timing
start = time.time()
step = 0

def current_game_state(info_state):
    return np.array(list(info_state.values()))

state = env.reset()
# Return values for RAM info into np array
game_state = current_game_state(env.state_info)

def run_initial_sequence():
    # select a quest
    env.pressbutton('NOOP')
    env.pressbutton('A')
    env.pressbutton('A')

    # select a name
    env.pressbutton('A')
    env.pressbutton('A')

    steps = 7  # not optimized
    for step in range(steps):
        env.pressbutton('right')
        env.pressbutton('down')

    # select fast dialogue
    env.pressbutton('A')
    env.pressbutton('up')
    env.pressbutton('A')

    # advance through initial dialogue

    steps = 9  # not optimized
    for step in range(steps):
        env.pressbutton('A')
    env.pressbutton('B')

total_reward = 0
episode_frame = 0

def doaction(action, state=state, game_state=game_state, total_reward=total_reward, episode_frame=episode_frame, printtiming=False):

    next_state, reward, done, info = env.step(action=action)
    return action, next_state, reward, done, info, state, game_state, total_reward, episode_frame

def actionpostprocessing(action, next_state, reward, done, info, state=state, game_state=game_state, total_reward=total_reward, episode_frame=episode_frame, printtiming=True):

    now = datetime.datetime.now()
    next_game_state = current_game_state(env.state_info)
    if printtiming:
        print(f'duration to step environment forward: {datetime.datetime.now() - now}')
    # Remember transition
    now = datetime.datetime.now()
    agent.add(experience=(state, next_state, game_state, next_game_state, action, reward, done))
    if printtiming:
        print(f'duration to add transition to the agent: {datetime.datetime.now() - now}')
    # Update agent
    now = datetime.datetime.now()
    agent.learn()
    if printtiming:
        print(f'duration to learn: {datetime.datetime.now() - now}')

    # Total reward
    total_reward += reward
    if print_stats_per_action == True:
        print(np.round(total_reward, 4), dragon_warrior_comboactions[action],
              episode_frame, np.round(agent.eps_now, 4))
    if pause_after_action == True:
        input('press any key to advance')

    return next_state, next_game_state, total_reward, episode_frame

def w_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['up'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results

def a_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['left'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results

def s_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['down'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results

def d_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['right'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results

def stairs_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['stairs'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results

def door_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['door'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results

def take_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['take'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results
def b_press(printtiming=True):
    results = doaction(env.dict_comboactionsindextoname['B'], printtiming=printtiming)
    # episode_frame += 1
    env.render()
    return results

# %%

class HumanPresser:

    def __init__(self):
        self.results = None
        pass

    def doaction(self, action, state=state, game_state=game_state, total_reward=total_reward, episode_frame=episode_frame, printtiming=False):

        next_state, reward, done, info = env.step(action=action)
        return action, next_state, reward, done, info, state, game_state, total_reward, episode_frame

    @property
    def dopostprocessingformpreviousstep(self, printtiming=True):

        action, next_state, reward, done, info, state, game_state, total_reward, episode_frame = self.results
        now = datetime.datetime.now()
        next_game_state = current_game_state(env.state_info)
        if printtiming:
            print(f'duration to step environment forward: {datetime.datetime.now() - now}')
        # Remember transition
        now = datetime.datetime.now()
        agent.add(experience=(state, next_state, game_state, next_game_state, action, reward, done))
        if printtiming:
            print(f'duration to add transition to the agent: {datetime.datetime.now() - now}')
        # Update agent
        now = datetime.datetime.now()
        agent.learn()
        if printtiming:
            print(f'duration to learn: {datetime.datetime.now() - now}')

        # Total reward
        total_reward += reward
        if print_stats_per_action == True:
            print(np.round(total_reward, 4), dragon_warrior_comboactions[action],
                  episode_frame, np.round(agent.eps_now, 4))
        if pause_after_action == True:
            input('press any key to advance')

        return next_state, next_game_state, total_reward, episode_frame

    def w(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['up'], printtiming=printtiming)
        env.render()

    def a(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['left'], printtiming=printtiming)
        env.render()

    def s(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['down'], printtiming=printtiming)
        env.render()

    def d(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['right'], printtiming=printtiming)
        env.render()

    def c(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['stairs'], printtiming=printtiming)
        env.render()

    def r(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['door'], printtiming=printtiming)
        env.render()

    def e(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['take'], printtiming=printtiming)
        env.render()

    def b(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if laststepwassuccessful:
            self.dopostprocessingformpreviousstep
        self.results = self.doaction(env.dict_comboactionsindextoname['B'], printtiming=printtiming)
        env.render()

# %%
#
# s = HumanPresser()
#
# # %%
#
# env.pressbutton('A')
# env.pressbutton('left')
# env.pressbutton('right')
# env.pressbutton('up')
# env.pressbutton('down')
# env.pressbutton('B')
#
# # %%
#
# s.s()
# s.a()
# s.w()
#
# s.x()

# %%
#
# actionpostprocessing(env.dict_comboactionsindextoname['left']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['right']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['up']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['down']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['stairs']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['door']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['take']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['B']); episode_frame += 1
# env.render()
#
# # %%
#
# actionpostprocessing(env.dict_comboactionsindextoname['right']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['take']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['right']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['take']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['right']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['right']); episode_frame += 1
# actionpostprocessing(env.dict_comboactionsindextoname['right']); episode_frame += 1

# %%

# Main loop
for episode in range(episodes):

    # Reset env, returns screen values into np array
    state = env.reset()
    # Return values for RAM info into np array
    game_state = current_game_state(env.state_info)

    # Reward
    total_reward = 0

    # %% advance through naming

    run_initial_sequence()

    # Play
    for episode_frame in range(frames_per_episode):

        if renderflag == True:
            # Slows down learning by a factor of 3
            env.render()

        # Run agent
        action = agent.run(state=state, game_state=game_state, eps_method=eps_method,
                           eps_cos_frames=eps_cosine_method_frames_per_cycle)

        results = doaction(action, printtiming=True)

        action, next_state, reward, done, info, state, game_state, total_reward, episode_frame = results
        # Update state and game_state
        state, game_state, total_reward, _ = actionpostprocessing(*results)

        # If done break loop
        # if done or info['exit_throne_room']:
        #     break

    # todo change to average eps
    if eps_method == 'exp_decay':
        eps = agent.eps
    if eps_method == 'cosine':
        eps = agent.eps_now

    # Rewards
    rewards.append(total_reward / episode_frame)
    frames_in_episode.append(episode_frame)
    episode_number.append(episode)
    epsilon_at_end.append(eps)
    if info['throne_room_key'] == True:
        key_found_in_episode.append(1)
    if info['throne_room_key'] == False:
        key_found_in_episode.append(0)
    if info['throne_room_gold'] == True:
        gold_found_in_episode.append(1)
    if info['throne_room_gold'] == False:
        gold_found_in_episode.append(0)
    if info['throne_room_torch'] == True:
        torch_found_in_episode.append(1)
    if info['throne_room_torch'] == False:
        torch_found_in_episode.append(0)

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
                                 k=info['throne_room_key'],
                                 g=info['throne_room_gold'],
                                 t=info['throne_room_torch']))
        start = time.time()
        step = agent.step

episode_dict = {
    'episode': episode_number,
    'end_epsilon': epsilon_at_end,
    'frames': frames_in_episode,
    'ave_reward': rewards,
    'key_found': key_found_in_episode,
    'gold_found': gold_found_in_episode,
    'torch_found': torch_found_in_episode,
}
results = pd.DataFrame(episode_dict).set_index('episode')

if use_dumb_dw_env == True:
    results.to_csv(f'dumb_DW_Bot_results.csv')
else:
    results.to_csv(f'DW_Bot_results.csv')

# Save rewards
np.save('rewards.npy', rewards)

# %%

