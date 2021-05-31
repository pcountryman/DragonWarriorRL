import os
from nes_py import NESEnv
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))
game_name = 'dragon_warrior.nes'
game_path = os.path.join(package_directory, game_name)

class DragonWarriorEnv(NESEnv):
    '''An OpenAI Gym interface to the NES game Final Fantasy'''

    # the range of rewards for each step
    reward_range = (-15, 15)

    def __init__(self):
        '''Initialize a new Dragon Warrior environment'''
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
        # hero position on current map
        self._hero_map = 0
        self._hero_x_pos = 0
        self._hero_y_pos = 0
        # variable to train network at beginning
        self._throne_key_chest_x_pos = 6
        self._throne_key_chest_y_pos = 1
        self._throne_room_door_x_pos = 4
        self._throne_room_door_y_pos = 6
        self._hero_gold = 0
        self._herb_count = 0
        self._hero_torches = 0
        self.reset()
        self._start_menu = False
        self._skip_start_screen()

    def _skip_start_screen(self):
        '''Press and release start to skip the start screen'''
        while self._start_menu == False:
            self._frame_advance(8)
            self._frame_advance(0)
            self._is_start_menu()

    def _is_start_menu(self):
        '''Return a boolean value for start menu screen based on ram value'''
        if self.ram[0x000b] == 63:
            if self.ram[0x000c] == 115:
                if self.ram[0x000d] == 154:
                    self._start_menu = True
        else:
            pass

    # todo add init method to bypass naming and start menu

    def _is_busy(self):
        '''Return a boolean value if an action is being processed'''
        # any button can yield busy state, value of 0 indicates no action is being processed
        if self.ram[0x0047] != 0:
            return True
        else:
            return False

    # todo set current values as init methods, ensure reseting can't be used to game stat growth etc
    def _current_exp(self):
        '''Return the current experience value'''
        return self.ram[0x00BA] + 256*self.ram[0x00BB]

    def _current_gold(self):
        '''Return the current gold value'''
        return self.ram[0x00BC] + 256*self.ram[0x00BD]

    def _current_magic_keys(self):
        '''Return the current number of magic keys'''
        return self.ram[0x00BF]

    def _current_magic_herbs(self):
        '''Return the current number of magic herbs'''
        return self.ram[0x00C0]

    def _current_torches(self):
        '''Return the current number of torches'''
        return self.ram[0x00C1]

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

    @property
    def state_info(self):
        '''Return a dictionary with key value pairs for information for agent'''
        return dict(
            map_id = self._map_id(),
            enemy_terrain_pointer = self._current_enemy_terrain_pointer(),
            is_enemy = 1 if self._is_enemy() else 0,
            map_x_pos = self._current_map_x_pos(),
            map_y_pos = self._current_map_y_pos(),
            current_hp = 0 if self._start_menu else self._current_hp(),
            current_atk = self._current_atk_power(),
            current_def = self._current_def_power(),
            current_magic_keys = self._current_magic_keys(),
            current_gold = self._current_gold(),
            current_exp = self._current_exp(),
            current_magic_herbs = self._current_magic_herbs(),
            current_torches = self._current_torches(),
        )

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
        self._magic_keys = 0
        self._hero_gold = 0
        self._herb_count = 0
        self._start_menu = False
        self._skip_start_screen()
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

    @property
    def _leave_throne_room(self):
        '''Return a boolean determining if the bot exited the throne room'''
        # Tantagel Castle is 4 on map id
        return self._map_id() == 4

    @property
    def _get_throne_room_torch(self):
        '''Return a boolean determining if the bot got the torch in the throne room'''
        return self._current_torches() == 1

    @property
    def _get_throne_room_gold(self):
        '''Return a boolean determining if the bot got the gold in the throne room'''
        return self._current_gold() > 0

    @property
    def _get_throne_room_key(self):
        '''Return a boolean determining if the bot got the key in the throne room'''
        return self._magic_keys > 0

    def _death_penalty(self):
        '''Return the reward for dying'''
        if self._is_dead:
            _reward = -25
        else:
            _reward = 0
        return _reward

    def _exp_reward(self):
        # return the reward based on party experience gained
        _reward = (self._current_exp() - self._hero_exp) / (self._hero_exp + 0.0001) * 100

        # do not reward paltry experience gains of 2% or less of hero exp
        if (_reward / (self._hero_exp + 0.00001)) < 2:
            _reward = 0

        # determine new hero exp value from previous exp values
        self._hero_exp = self._current_exp()

        return _reward

    def _gold_reward(self):
        '''Return the reward based on gold acquired'''
        _reward = (self._current_gold() - self._hero_gold) / (self._hero_gold + 0.0001) * 100
        self._hero_gold = self._current_gold()
        return _reward

    # todo refine this for HP recovery once battle are possible
    def _herb_reward(self):
        '''Return the reward for gaining herbs'''
        if self._current_magic_herbs() > self._herb_count:
            _reward = 15
        else:
            _reward = 0
        self._herb_count = self._current_magic_herbs()
        return _reward

    # todo refine this for dungeon use
    def _torch_reward(self):
        '''Return the reward for gaining torches'''
        if self._current_torches() > self._hero_torches:
            _reward = 5
        else:
            _reward = 0
        self._hero_torches = self._current_torches()
        return _reward

    def _open_door_reward(self):
        '''Return the reward for opening a door using a magic key.
        The only way to get rid of a key is to use it.'''
        if self._current_magic_keys() < self._magic_keys:
            _reward = 15
        else:
            _reward = 0
        # Update key value
        self._magic_keys = self._current_magic_keys()
        return _reward

    def _gain_magic_key_reward(self):
        '''Return the reward for acquiring a magic key'''
        if self._current_magic_keys() > self._magic_keys:
            _reward = 15
        else:
            _reward = 0
        # Update key value
        self._magic_keys = self._current_magic_keys()
        return _reward

    # first attempt to encourage exploration
    def _is_same_pos(self):
        '''Return boolean operator if hero has not moved'''
        if self._hero_x_pos == self._current_map_x_pos():
            if self._hero_y_pos == self._current_map_y_pos():
                if self._hero_map == self._map_id():
                    return True
        else:
            return False

    # stationary penalty
    def _stationary_penalty(self):
        if self._is_same_pos():
            _reward = -0.1
        else:
            _reward = 0
        self._hero_x_pos = self._current_map_x_pos()
        self._hero_y_pos = self._current_map_y_pos()
        self._hero_map = self._map_id()
        return _reward

    # todo use 0x0096 value FF and 0x0097 value 0C for Command menu for directional actions

    # not used,
    def _throne_room_key_reward(self):
        if self._magic_keys == 0:
            key_chest_penalty = np.sqrt((self._current_map_y_pos() - self._throne_key_chest_y_pos) ** 2 +
                       (self._current_map_x_pos() - self._throne_key_chest_x_pos) ** 2)
            if key_chest_penalty > 0:
                _reward = -0.001
            else:
                _reward = 1
        else:
            door_open_penalty = np.sqrt((self._current_map_y_pos() - self._throne_room_door_y_pos) ** 2 +
                       (self._current_map_x_pos() - self._throne_room_door_x_pos) ** 2)
            if door_open_penalty > 0:
                _reward = -0.001
            else:
                _reward = 1
        return _reward

    # use this method to assure each action is resolved on nes-py
    def _frame_buffering(self):
        # A button input lag
        while self._is_busy():
            self.frame_advance(0)
        # for i in range(2):
        #     self._frame_advance(0)

    def frame_advance(self, action):
        '''Return the frame advance function'''
        return self._frame_advance(action)

    def _get_reward(self):
        '''Return the reward after a step occurs'''
        self._frame_buffering()
        return (self._exp_reward() + self._open_door_reward() + self._gain_magic_key_reward() +
                self._herb_reward() + self._gold_reward() +
                self._torch_reward() + self._stationary_penalty())

    def _get_done(self):
        '''Return True if the episode is over, False otherwise'''
        return False

    # include RAM info needed for agent (map_id, hero_stats, map_position, etc)
    def _get_info(self):
        '''Return the info after a step occurs'''
        return dict(
            exit_throne_room = self._leave_throne_room,
            throne_room_gold = self._get_throne_room_gold,
            throne_room_torch = self._get_throne_room_torch,
            throne_room_key = self._get_throne_room_key
        )


# explicitly define the outward facing API for the module
__all__ = [DragonWarriorEnv.__name__]