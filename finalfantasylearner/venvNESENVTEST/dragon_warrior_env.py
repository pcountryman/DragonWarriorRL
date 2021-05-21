import os
from nes_py import NESEnv
import numpy as np
import pandas as pd

package_directory = os.path.dirname(os.path.abspath(__file__))

game_path = 'dragon_warrior.nes'

class DragonWarriorEnv(NESEnv):
    '''An OpenAI Gym interface to the NES game Final Fantasy'''

    # the range of rewards for each step
    reward_range = (-15,15)

    def __init__(self):
        '''Initialize a new Final Fantasy environment'''
        super(DragonWarriorEnv, self).__init__(game_path)
        # setup any variables to use in the below callbacks here

        # setup a variable to keep track of party exp
        self._hero_exp = 0
        # setup a variable to keep track of number of magic keys
        self._magic_keys = 0
        # initial position of sprites on throne room map
        self._current_map = 5
        self._map_x_pos_last = 3
        self._map_y_pos_last = 4
        # variable to train network at beginning
        self._throne_key_chest_x_pos = 6
        self._throne_key_chest_y_pos = 1
        self._throne_room_door_x_pos = 4
        self._throne_room_door_y_pos = 6

    def _current_exp(self):
        '''Return the current experience value'''
        return self.ram[0x00BA]

    def _current_hp(self):
        '''Return the current hit point value'''
        return self.ram[0x00C5]

    def _current_atk_power(self):
        '''Return the current atk power, adjusted to include equipment stats'''
        return self.ram[0x00CC]

    def _current_def_power(self):
        '''Return the current def power, adjusted to include equipment stats'''
        return self.ram[0x00CD]

    def _current_map_x_pos(self):
        '''Return the current x position on the map'''
        return self.ram[0x003A]

    def _current_map_y_pos(self):
        '''Return the current y position on the map'''
        return self.ram[0x003B]

    def _overworld_x_pos(self):
        '''Return the current overworld x position'''
        return self.ram[0x0042]

    def _overworld_y_pos(self):
        '''Return the current overworld y position'''
        return self.ram[0x003E]

    def _music_tempo(self):
        '''Return the tempo of music'''
        return self.ram[0x00FB]

    def _is_enemy(self):
        '''Determines if enemy_terrain_pointer is pointint to enemy or terrain based on the music being played'''
        if self._music_tempo() == 120:
            return True
        else:
            return False

    def _current_enemy_terrain_pointer(self):
        '''Return what enemy you are facing in battle, or what terrain you are standing in'''
        return self.ram[0x00e0]

    def _map_id(self):
        '''Return the current map id'''
        return self.ram[0x0045]

    def _current_magic_keys(self):
        '''Return the current number of magic keys'''
        return self.ram[0x00BF]

    def _will_reset(self):
        '''Handle any RAM hacking before a reset occurs'''
        # use this method to perform setup before an episode resets.
        # the method returns None

        pass

    def _did_reset(self):
        '''Handle any RAM hacking after a reset occurs'''
        # use this method to access the RAM of the emulator
        # and perform setup for each episode
        # the method returns None
        # todo add in method that automatically selects continue at title screen
        # todo add in method that stores initial x and y pos on worldmap
        pass

    def _did_step(self, done):
        '''
        Handle any RAM hacking after a step occurs.
        Args:
            done: whether the done flag is set to true

        Returns:
            None

        '''
        pass

    '''Simple death penalty'''
    def _is_dead(self):
        '''Return if the entire party is dead'''
        if self._current_hp() == 0:
            return True
        else:
            return False

    def _death_penalty(self):
        '''Return the reward for dying'''
        if self._is_dead:
            return -25
        return 0

    def _exp_reward(self):
        # return the reward based on party experience gained
        _reward = self._current_exp() - self._hero_exp

        # do not reward paltry experience gains of 2% or less of hero exp
        if (_reward/(self._hero_exp + 0.00001)) < 0.02:
            return 0

        # determine new hero exp value from previous exp values
        self._hero_exp = self._current_exp()

        return _reward

    def _open_door_reward(self):
        '''Return the reward for opening a door using a magic key.
        The only way to get rid of a key is to use it.'''
        if self._current_magic_keys() < self._magic_keys:
            _reward = 5
        else:
            _reward = 0
        # Update key value
        self._magic_keys = self._current_magic_keys()
        return _reward
    
    def _gain_magic_key_reward(self):
        '''Return the reward for acquiring a magic key'''
        if self._current_magic_keys() > self._magic_keys:
            _reward = 5
        else:
            _reward = 0
        # Update key value
        self._magic_keys = self._current_magic_keys()
        return _reward

    def _throne_room_key_reward(self):
        if self._magic_keys == 0:
            if np.sqrt((self._current_map_y_pos() - self._throne_key_chest_y_pos) ** 2 +
                       (self._current_map_x_pos() - self._throne_key_chest_x_pos) ** 2) > 0:
                _reward = -1
            else:
                _reward = 0
        else:
            if np.sqrt((self._current_map_y_pos() - self._throne_room_door_y_pos) ** 2 +
                       (self._current_map_x_pos() - self._throne_room_door_x_pos) ** 2) > 0:
                _reward = -1
            else:
                _reward = 0
        return _reward



    def _get_reward(self):
        '''Return the reward after a step occurs'''
        return (self._exp_reward() + self._open_door_reward() + self._gain_magic_key_reward() +
                self._throne_room_key_reward())

    def _get_done(self):
        '''Return True if the episode is over, False otherwise'''
        return False

    def _get_info(self):
        '''Return the info after a step occurs'''
        return {}

# explicitly define the outward facing API for the module
__all__ = [DragonWarriorEnv.__name__]