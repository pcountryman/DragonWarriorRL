import datetime

import numpy as np


def current_game_state(info_state):
    return np.array(list(info_state.values()))


class Actor:
    '''
    The actor groups many of the Q-learning steps into a few methods that make it more convenient to call those
    frequently strung together steps. It allows these groups of steps to be shared in the bot controlled and human
    controlled scripts, which is intended to make it easier to maintain both systems.
    '''

    def __init__(self, env, state, game_state, total_reward, agent, print_stats_per_action,
                 dragon_warrior_comboactions, pause_after_action, _episode_frame=0):
        self.env = env
        self.state = state
        self.game_state = game_state
        self.total_reward = total_reward
        self.agent = agent
        self.print_stats_per_action = print_stats_per_action
        self.dragon_warrior_comboactions = dragon_warrior_comboactions
        self.pause_after_action = pause_after_action
        self._episode_frame = _episode_frame

        self._results = None

    # Episode frame is incremented by the for loop in run_dragon_warrior, but I want that information here.
    # To handle that, we implemented a setter function to understand how to update the property, but to allow
    # the same interface for interacting with episode_frame as the attributes that are passed in at initialization.
    @property
    def episode_frame(self):
        return self._episode_frame

    @episode_frame.setter
    def episode_frame(self, value):
        self._episode_frame = value

    @property
    def game_state(self):
        return self._game_state

    @game_state.setter
    def game_state(self, value):
        self._game_state = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    def doaction(self, action):
        '''Executes the action and returns the new state and reward.'''
        # next_state, reward, done, info = self.env.step(action=action)
        next_state, reward, done, info = self.env.step(action=action)
        self.results = action, next_state, reward, done, info

        # return self.results

    def dopostprocessingformpreviousstep(self, printtiming=True):
        '''Gets the game state, adds the last iteration to the Q-table, trains the neural network, and updates the
        reward.'''

        action, next_state, reward, done, info = self.results
        now = datetime.datetime.now()
        next_game_state = current_game_state(self.env.state_info)
        if printtiming:
            print(f'duration to step environment forward: {datetime.datetime.now() - now}')
        # Remember transition
        now = datetime.datetime.now()
        self.agent.add(experience=(self.state, next_state, self.game_state, next_game_state, action, reward, done))
        if printtiming:
            print(f'duration to add transition to the agent: {datetime.datetime.now() - now}')
        # Update agent
        now = datetime.datetime.now()
        self.agent.learn()
        if printtiming:
            print(f'duration to learn: {datetime.datetime.now() - now}')

        # Total reward
        self.total_reward += reward
        if self.print_stats_per_action == True:
            print(np.round(self.total_reward, 4), self.dragon_warrior_comboactions[action],
                  self.episode_frame, np.round(self.agent.eps_now, 4))
        if self.pause_after_action == True:
            input('press any key to advance')

        self.game_state = next_game_state
        self.state = next_state

# %%

