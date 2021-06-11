import pathlib
import time

import numpy as np
import pandas as pd

# from dumb_dw_env import DumbDragonWarriorEnv
from actions import dragon_warrior_actions, dragon_warrior_comboactions
from dqn_agent import DQNAgent
from dragon_warrior_env import DragonWarriorEnv
from environmentwrappers import ButtonRemapper
from utilities import current_game_state, Actor

######

# parameters for arctansin eps function that spends more time at boundary limits
# and thus spends more time in chaos (high eps) or order (low eps)
arctansin_yint = 0.5  # adjusts y intercept
arctansin_amp = 0.48  # adjusts amplitude
arctansin_freq = 0.01  # completes one cycle every 1/c steps, roughly half at high eps and half at low eps
# todo make arctansin_freq reward velocity dependent
arctansin_delta = 0.1  # adjusts function sharpness, how quickly between 0 and 1

episodes = 20
frames_per_episode = 500
loadcheckpoint = True
# loadcheckpoint = False
# renderflag = False
renderflag = True
# print_stats_per_action = True
print_stats_per_action = False
pause_after_action = False
# pause_after_action = True
# printtiming = True
printtiming = False

use_dumb_dw_env = False
# todo adjust cosine method, or change from boolean to variable controlling period of cosine
eps_method = 'arctansin'
# eps_method = 'cosine'
# eps_method = 'exp_decay'
eps_cosine_method_frames_per_cycle = 500  # travels one wavelength in this
frames_to_elapse_before_saving_agent = 2000
path_models = pathlib.Path('models/')
filename_model = str(path_models / 'model')
######

# create list of only comboactions
flatten_comboactions = [j for sub in dragon_warrior_comboactions for j in sub]
flatten_actions = [j for sub in dragon_warrior_actions for j in sub]
only_comboactions = [x for x in flatten_comboactions if x not in flatten_actions]

if use_dumb_dw_env:
    env = DumbDragonWarriorEnv()
else:
    env = DragonWarriorEnv(comboactionsonly=only_comboactions)
env = ButtonRemapper(env, dragon_warrior_actions, dragon_warrior_comboactions, renderflag=renderflag)

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
                 double_q=True, arctansin_yint=arctansin_yint, arctansin_amp=arctansin_amp,
                 arctansin_freq=arctansin_freq, arctansin_delta=arctansin_delta,
                 eps_method=eps_method, agent_save=frames_to_elapse_before_saving_agent)
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
exit_throne_room_in_episode = []
exit_throne_room_permanent_in_episode = []
found_fairy_flute_in_episode = []
hero_exp_in_episode = []
hero_gold_in_episode = []
hero_final_map_id_in_episode = []
hero_final_xpos_in_episode = []
hero_final_ypos_in_episode = []
hero_final_weapon_in_episode = []
hero_final_armor_in_episode = []
hero_final_shield_in_episode = []

# Timing
start = time.time()
step = 0
episode_frame = 0

# Return values for RAM info into np array
game_state = current_game_state(env.state_info)

# Main loop
for episode in range(episodes):

    # Reset env, returns screen values into np array
    state = env.reset()
    # Return values for RAM info into np array
    game_state = current_game_state(env.state_info)

    # Reward
    total_reward = 0

    actor = Actor(env, state, game_state, total_reward, agent, print_stats_per_action,
                  dragon_warrior_comboactions, pause_after_action)

    # Play
    for episode_frame in range(frames_per_episode):
        actor.episode_frame = episode_frame

        if renderflag:
            # Slows down learning by a factor of 3
            env.render()

        # Run agent
        action = agent.run(state=state, game_state=game_state, eps_method=eps_method,
                           eps_cos_frames=eps_cosine_method_frames_per_cycle)

        actor.doaction(action)

        # Place action in env so it can be used for reward evaluation
        # Cannot use env.herocurrentcomboaction as that only writes the instance variable
        DragonWarriorEnv.herocurrentcomboaction = dragon_warrior_comboactions[action][0]

        # put everything into results for convenience, but some of the components are needed elsewhere so
        # breaking it back out here so it can more easily consumed and inspected.
        action, next_state, reward, done, info = actor.results

        # Update state and game_state
        if printtiming:
            actor.dopostprocessingfrompreviousstep()
        else:
            actor.dopostprocessingfrompreviousstep(printtiming=printtiming)
        # pull out state information so we can pass it into the agent in the next iteration.
        game_state = actor.game_state
        state = actor.state

        # If done break loop
        # if done or info['exit_throne_room']:
        #     break

    # todo change to average eps
    if eps_method == 'exp_decay':
        eps = agent.eps
    if eps_method in ['cosine', 'arctansin']:
        eps = agent.eps_now
    else:
        eps = 0

    # Rewards
    total_reward = actor.total_reward
    rewards.append(total_reward / episode_frame)
    frames_in_episode.append(episode_frame)
    episode_number.append(episode)
    epsilon_at_end.append(eps)
    if info['throne_room_key']:
        key_found_in_episode.append(1)
    if not info['throne_room_key']:
        key_found_in_episode.append(0)
    if info['throne_room_gold']:
        gold_found_in_episode.append(1)
    if not info['throne_room_gold']:
        gold_found_in_episode.append(0)
    if info['throne_room_torch']:
        torch_found_in_episode.append(1)
    if not info['throne_room_torch']:
        torch_found_in_episode.append(0)
    if info['exit_throne_room']:
        exit_throne_room_in_episode.append(1)
    if not info['exit_throne_room']:
        exit_throne_room_in_episode.append(0)
    if info['exit_throne_room_perm']:
        exit_throne_room_permanent_in_episode.append(1)
    if not info['exit_throne_room_perm']:
        exit_throne_room_permanent_in_episode.append(0)
    if info['found_fairy_flute']:
        found_fairy_flute_in_episode.append(1)
    if not info['found_fairy_flute']:
        found_fairy_flute_in_episode.append(0)
    hero_exp_in_episode.append(info['hero_exp'])
    hero_gold_in_episode.append(info['hero_gold'])
    hero_final_map_id_in_episode.append(info['hero_map_id'])
    hero_final_xpos_in_episode.append(info['hero_xpos'])
    hero_final_ypos_in_episode.append(info['hero_ypos'])
    hero_final_weapon_in_episode.append(info['hero_weapon'])
    hero_final_armor_in_episode.append(info['hero_armor'])
    hero_final_shield_in_episode.append(info['hero_shield'])

    # todo build lists/dictionaries with this info, export as csv
    if episode % 1 == 0:
        print('Episode {e} - +'
              'Frame {f} - +'
              'Frames/sec {fs} - +'
              'Mean reward {r} - +'
              'Total reward {R} - +'
              'Key {k} - +'
              'Gold {g} - +'
              'Escape {esc}'.format(e=episode,
                                    f=agent.step,
                                    fs=np.round((agent.step - step) / (time.time() - start)),
                                    r=np.round(rewards[-1:], 4),
                                    R=np.round(total_reward, 4),
                                    k=info['throne_room_key'],
                                    g=info['throne_room_gold'],
                                    esc=info['exit_throne_room']))
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
    'exit_throne_room': exit_throne_room_in_episode,
    'fairy_flute_found': found_fairy_flute_in_episode,
    'hero_exp': hero_exp_in_episode,
    'hero_gold': hero_gold_in_episode,
    'hero_final_map': hero_final_map_id_in_episode,
    'hero_final_xpos': hero_final_xpos_in_episode,
    'hero_final_ypos': hero_final_ypos_in_episode,
    'hero_final_weapon': hero_final_weapon_in_episode,
    'hero_final_armor': hero_final_armor_in_episode,
    'hero_final_shield': hero_final_shield_in_episode,
}

results = pd.DataFrame(episode_dict).set_index('episode')

if use_dumb_dw_env:
    results.to_csv(f'dumb_DW_Bot_results.csv')
else:
    results.to_csv(f'DW_Bot_results.csv')

# Save rewards
np.save('rewards.npy', rewards)

# %%
