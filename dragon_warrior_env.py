import math
import os
from nes_py import NESEnv

package_directory = os.path.dirname(os.path.abspath(__file__))
game_name = 'dragon_warrior.nes'
game_path = os.path.join(package_directory, game_name)


class DragonWarriorEnv(NESEnv):
    """An OpenAI Gym interface to the NES game Final Fantasy"""

    # the range of rewards for each step
    reward_range = (-1, 1)

    # setup variable to store current hero action
    herocurrentcomboaction = None

    def __init__(self, comboactionsonly):
        """Initialize a new Dragon Warrior environment"""
        super(DragonWarriorEnv, self).__init__(game_path)
        # setup any variables to use in the below callbacks here

        # setup a variables to keep track of hero statistics
        self._hero_exp = 0
        self._hero_gold = 0
        self._hero_hp = 0
        self._hero_mp = 0
        self._hero_lvl = 0
        self._hero_str = 0
        self._hero_agi = 0
        self._hero_max_hp = 0
        self._hero_max_mp = 0
        self._hero_atk_power = 0
        self._hero_def_power = 0
        self._hero_herb_count = 0
        self._hero_torch_count = 0
        self._hero_magic_keys = 0
        self._hero_weapon = 0
        self._hero_armor = 0
        self._hero_shield = 0
        # initial position of hero sprite on throne room map
        self._current_map = 5
        self._map_x_pos_last = 3
        self._map_y_pos_last = 4
        # hero position on current map
        self._hero_map_id = 0
        self._hero_x_pos = 0
        self._hero_y_pos = 0
        # variable to train network at beginning
        self._throne_key_chest_x_pos = 6
        self._throne_key_chest_y_pos = 1
        self._throne_room_door_x_pos = 4
        self._throne_room_door_y_pos = 6
        self._throne_room_stairs_xpos = 8
        self._throne_room_stairs_ypos = 8
        # flags to bypass start menu, hero creation, and hero selection
        self._start_menu = False
        self._quest_menu = False
        self._adventure_log_menu = False
        self._respawn_in_throne_room = False
        self._naming_menu = False
        self._normal_dialogue_speed = False
        self._fast_dialogue_speed = False
        self._continue_quest_menu = False
        self._save_quest_menu = False
        # flags to monitor hero progress through the game
        self._pick_up_throne_room_torch = False
        self._pick_up_throne_room_torch_lose_on_reset = False
        self._pick_up_throne_room_gold = False
        self._pick_up_throne_room_gold_lose_on_reset = False
        self._pick_up_throne_room_key = False
        self._pick_up_throne_room_key_lose_on_reset = False
        self._escape_throne_room = False
        self._escape_throne_room_lose_on_reset = False
        self._obtain_fairy_flute = False
        self._obtain_fairy_flute_lose_on_reset = False
        self._obtain_stones_of_sunlight = False
        self._obtain_stones_of_sunlight_lose_on_reset = False
        self._obtain_staff_of_rain = False
        self._obtain_staff_of_rain_lose_on_reset = False
        self._obtain_rainbow_drop = False
        self._obtain_rainbow_drop_lose_on_reset = False
        self._obtain_erdrick_armor = False
        self._obtain_erdrick_armor_lose_on_reset = False
        self._obtain_erdrick_sword = False
        self._obtain_erdrick_sword_lose_on_reset = False
        # initialize ROM
        self.reset()
        self._dragon_warrior_combo_actions = comboactionsonly

    # %%
    # define functions/methods that bypass start menu, hero creation, and hero selection
    def _skip_start_screen(self):
        """Press and release start to skip the start screen"""
        while not self._start_menu:
            self._frame_advance(8)
            self._frame_advance(0)
            self._is_start_menu()

    def _skip_main_menu(self):
        """Press and release A to skip the main menu screen"""
        while not self._quest_menu:
            self._frame_advance(1)
            self._frame_advance(0)
            self._is_quest_menu()

    def _skip_adventurer_log_menu(self):
        """Press and release A to skip the adventurer log screen"""
        while not self._adventure_log_menu:
            self._frame_advance(1)
            self._frame_advance(0)
            self._is_adventure_log_menu()

    def _skip_to_respawn(self):
        """Press and release A to skip to the throne room screen"""
        while not self._respawn_in_throne_room:
            self._frame_advance(1)
            self._frame_advance(0)
            self._is_in_throne_room()

    def _is_in_throne_room(self):
        """Return a boolean value based on RAM values for hero in throne room"""
        if self._current_map_id() == 5:
            self._respawn_in_throne_room = True
        else:
            pass

    def _is_adventure_log_menu(self):
        """Return a boolean value based on RAM values for menu selection at start of game"""
        if self.ram[0x000a] in [200, 140]:
            if self.ram[0x000c] == 115:
                if self.ram[0x000d] == 154:
                    self._adventure_log_menu = True
        else:
            pass

    def _is_quest_menu(self):
        """Return a boolean value based on RAM values for menu selection at start of game"""
        if self.ram[0x000a] == 70:
            if self.ram[0x000b] == 33:
                if self.ram[0x000c] == 115:
                    if self.ram[0x000d] == 154:
                        self._quest_menu = True
        else:
            pass

    def _is_start_menu(self):
        """Return a boolean value for start menu screen based on ram value"""
        if self.ram[0x000b] == 63:
            if self.ram[0x000c] == 115:
                if self.ram[0x000d] == 154:
                    self._start_menu = True
        else:
            pass

    def _is_save_continue_quest_menu(self):
        """Return a boolean value for save menu"""
        if self._hero_map_id == 5:  # if in throne room
            if self._hero_x_pos == 3:
                if self._hero_y_pos == 4:  # standing before king (can be multiple positions)
                    if self.ram[0x000a] == 130:
                        if self.ram[0x000b] == 35:
                            if self.ram[0x000c] == 86:
                                if self.ram[0x000d] == 154:
                                    self._continue_quest_menu = True
        else:
            pass

    # %%
    # Assorted functions/methods
    def _is_busy(self):
        """Return a boolean value if an action is being processed"""
        # any button can yield busy state, value of 0 indicates no action is being processed
        if self.ram[0x0047] != 0:
            return True
        else:
            return False

    @property
    def state_info(self):
        """Return a dictionary with key value pairs for information for agent"""
        return dict(
            map_id=self._current_map_id(),
            enemy_terrain_pointer=self._current_enemy_terrain_pointer(),
            is_enemy=1 if self._is_enemy() else 0,
            map_x_pos=self._current_map_x_pos(),
            map_y_pos=self._current_map_y_pos(),
            current_hp=0 if self._start_menu else self._current_hp(),
            current_atk=self._current_atk_power(),
            current_def=self._current_def_power(),
            current_magic_keys=self._current_magic_keys(),
            current_gold=self._current_gold(),
            current_exp=self._current_exp(),
            current_magic_herbs=self._current_magic_herbs(),
            current_torches=self._current_torches(),
        )

    def is_command_window_open(self):
        if self.command_window_state() == 255:
            return True
        else:
            return False

    # use this method to assure each action is resolved on nes-py
    def _frame_buffering(self):
        # A button input lag
        while self._is_busy():
            self.frame_advance(0)

    def frame_advance(self, action):
        """Return the frame advance function"""
        return self._frame_advance(action)

    # %%
    # functions/methods that retrieve values from the emulator RAM
    # todo set current values as init methods, ensure reseting can't be used to game stat growth etc

    # todo use 0x0096 value FF and 0x0097 value 0C for Command menu for directional actions
    def command_window_state(self):
        """Return the hex value from the location of the command window state."""
        return self.ram[0x0096]

    def _current_exp(self):
        """Return the current experience value"""
        return self.ram[0x00BA] + 256 * self.ram[0x00BB]

    def _current_gold(self):
        """Return the current gold value"""
        return self.ram[0x00BC] + 256 * self.ram[0x00BD]

    def _current_hp(self):
        """Return the current hp value"""
        return self.ram[0x00C5]

    def _current_mp(self):
        """Return the current mp value"""
        return self.ram[0x00C6]

    def _current_lvl(self):
        """Return the current lvl value"""
        return self.ram[0x00C7]

    def _current_str(self):
        """Return the current str value"""
        return self.ram[0x00C8]

    def _current_agi(self):
        """Return the current agi value"""
        return self.ram[0x00C9]

    def _current_max_hp(self):
        """Return the current maximum hp value"""
        return self.ram[0x00CA]

    def _current_max_mp(self):
        """Return the current maximum mp value"""
        return self.ram[0x00CB]

    def _current_atk_power(self):
        """Return the current atk power value"""
        return self.ram[0x00CC]

    def _current_def_power(self):
        """Return the current def power value"""
        return self.ram[0x00CD]

    def _current_magic_keys(self):
        """Return the current number of magic keys"""
        return self.ram[0x00BF]

    def _current_magic_herbs(self):
        """Return the current number of magic herbs"""
        return self.ram[0x00C0]

    def _current_torches(self):
        """Return the current number of torches"""
        return self.ram[0x00C1]

    def _current_map_x_pos(self):
        """Return the current x position on the map"""
        return self.ram[0x003A]

    def _current_map_y_pos(self):
        """Return the current y position on the map"""
        return self.ram[0x003B]

    def _overworld_x_pos(self):
        """Return the current overworld x position"""
        return self.ram[0x0042]

    def _overworld_y_pos(self):
        """Return the current overworld y position"""
        return self.ram[0x003E]

    def _music_tempo(self):
        """Return the tempo of music"""
        return self.ram[0x00FB]

    def _is_enemy(self):
        """Determines if enemy_terrain_pointer is pointint to enemy or terrain based on the music being played"""
        if self._music_tempo() == 120:
            return True
        else:
            return False

    def _current_enemy_terrain_pointer(self):
        """Return what enemy you are facing in battle, or what terrain you are standing in"""
        return self.ram[0x00e0]

    def _current_map_id(self):
        """Return the current map id"""
        return self.ram[0x0045]

    def _current_equipment(self):
        """Return the hex value for hero equipment"""
        return self.ram[0x00BE]

    def _current_weapon(self):
        """Return what weapon the hero has"""
        weapon = hex(self._current_equipment())[-2:-1]  # weapons stored in first number value in hex
        if weapon == '2':
            return 'bamboo pole'
        if weapon == '4':
            return 'club'
        if weapon == '6':
            return 'copper sword'
        if weapon == '8':
            return 'hand axe'
        if weapon == 'a':
            return 'broad sword'
        if weapon == 'c':
            return 'flame sword'
        if weapon == 'e':
            return 'erdrick sword'
        else:
            return 'no weapon'

    def _current_shield(self):
        """Return the current armor the hero has"""
        armor_and_shield = self._current_equipment() % 32  # weapon values are every 32 in hex
        shield = armor_and_shield % 4  # armor values are every 4 in hex
        if shield == 0:
            return 'no shield'
        if shield == 1:
            return 'small shield'
        if shield == 2:
            return 'large shield'
        if shield == 3:
            return 'silver shield'

    def _current_armor(self):
        """Return the current armor the hero has"""
        armor_and_shield = self._current_equipment() % 32  # weapon values are every 32 in hex
        shield = armor_and_shield % 4  # armor values are every 4 in hex
        armor = armor_and_shield - shield
        if armor == 0:
            return 'no armor'
        if armor == 4:
            return 'clothes'
        if armor == 8:
            return 'leather armor'
        if armor == 12:
            return 'chain mail'
        if armor == 16:
            return 'half plate armor'
        if armor == 20:
            return 'full plate armor'
        if armor == 24:
            return 'magic armor'
        if armor == 28:
            return 'erdrick armor'

    # %%
    # functions/methods that overwrite functions/methods from the nes-py emulator

    def _will_reset(self):
        """Handle any RAM hacking before a reset occurs"""
        # use this method to perform setup before an episode resets.
        # the method returns None
        # todo flesh out reset loss methodology more so it includes rewards gained

        pass

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs"""
        # use this method to access the RAM of the emulator
        # and perform setup for each episode
        # the method returns None
        self._current_map = 5
        self._map_x_pos_last = 3
        self._map_y_pos_last = 4
        self._start_menu = False
        self._quest_menu = False
        self._adventure_log_menu = False
        self._respawn_in_throne_room = False
        self._skip_start_screen()
        self._skip_main_menu()
        self._skip_adventurer_log_menu()
        self._skip_to_respawn()
        self._pick_up_throne_room_torch_lose_on_reset = False
        self._pick_up_throne_room_gold_lose_on_reset = False
        self._pick_up_throne_room_key_lose_on_reset = False
        self._escape_throne_room_lose_on_reset = False
        self._obtain_fairy_flute_lose_on_reset = False
        self._obtain_stones_of_sunlight_lose_on_reset = False
        self._obtain_staff_of_rain_lose_on_reset = False
        self._obtain_rainbow_drop_lose_on_reset = False
        self._obtain_erdrick_armor_lose_on_reset = False
        self._obtain_erdrick_sword_lose_on_reset = False
        pass

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.
        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        pass

    # %%
    # functions/methods that return boolean values based on action taken by agent

    def _skip_continue_quest_menu(self):
        """Force the hero to save when they arrive at save menu option"""
        if self._is_save_continue_quest_menu():
            self.frame_advance(1)
            self.frame_advance(0)

    def _did_hero_just_save(self):
        """Force the hero to save when they arrive at save menu option"""
        if self._is_save_continue_quest_menu():
            self._skip_continue_quest_menu()
            return True
        else:
            return False

    def _is_dead(self):
        """Return if the hero is dead"""
        if self._current_hp() == 0:
            return True
        else:
            return False

    def _is_fairy_flute_in_inventory(self):
        """Return true if the fairy flute is in hero inventory"""
        if '5' in hex(self.ram[0x00C1]):
            return True
        else:
            return False

    def _did_hp_increase(self):
        if self._current_hp() > self._hero_hp:
            return True
        else:
            return False

    def _did_hp_decrease(self):
        if self._current_hp() < self._hero_hp:
            return True
        else:
            return False

    def _did_mp_increase(self):
        if self._current_mp() > self._hero_mp:
            return True
        else:
            return False

    def _did_mp_decrease(self):
        if self._current_mp() < self._hero_mp:
            return True
        else:
            return False

    def _did_lvl_increase(self):
        if self._current_lvl() > self._hero_lvl:
            return True
        else:
            return False

    def _did_str_increase(self):
        if self._current_str() > self._hero_str:
            return True
        else:
            return False

    def _did_str_decrease(self):
        if self._current_str() < self._hero_str:
            return True
        else:
            return False

    def _did_agi_increase(self):
        if self._current_agi() > self._hero_agi:
            return True
        else:
            return False

    def _did_agi_decrease(self):
        if self._current_agi() < self._hero_agi:
            return True
        else:
            return False

    def _did_max_hp_increase(self):
        if self._current_max_hp() > self._hero_max_hp:
            return True
        else:
            return False

    def _did_max_mp_increase(self):
        if self._current_max_mp() > self._hero_max_mp:
            return True
        else:
            return False

    def _did_atk_power_increase(self):
        if self._current_atk_power() > self._hero_atk_power:
            return True
        else:
            return False

    def _did_atk_power_decrease(self):
        if self._current_atk_power() < self._hero_atk_power:
            return True
        else:
            return False

    def _did_def_power_increase(self):
        if self._current_def_power() > self._hero_def_power:
            return True
        else:
            return False

    def _did_def_power_decrease(self):
        if self._current_def_power() < self._hero_def_power:
            return True
        else:
            return False

    def _is_same_pos(self):
        """Return boolean operator if hero has not moved"""
        if self._hero_x_pos == self._current_map_x_pos():
            if self._hero_y_pos == self._current_map_y_pos():
                if self._hero_map_id == self._current_map_id():
                    return True
        else:
            return False

    def _is_same_map_id(self):
        """Return a boolean operator if hero has changed maps (or floors)"""
        if self._hero_map_id == self._current_map_id():
            return True
        else:
            return False

    # %%
    # functions/methods that calculate differences in state due to agent action
    def _diff_pos(self, pos1, pos2):
        """Return a value for the difference between two distances"""
        return pos1 - pos2

    # primitive distance comparison for goals
    def _distance_to_goal(self, goalxpos, goalypos):
        """Return a boolean value for hero progress towards a goal"""
        currentxpos = self._current_map_x_pos()
        currentypos = self._current_map_y_pos()
        currentdistance = math.sqrt(self._diff_pos(goalxpos, currentxpos) ** 2 +
                                    self._diff_pos(goalypos, currentypos) ** 2)
        return currentdistance

    # primitive distance comparison for goals
    def _is_hero_closer_to_goal(self, goalxpos, goalypos):
        """Return a boolean value for hero progress towards a goal"""
        currentxpos = self._current_map_x_pos()
        currentypos = self._current_map_y_pos()
        heroxpos = self._hero_x_pos
        heroypos = self._hero_y_pos
        currentdistance = math.sqrt(self._diff_pos(goalxpos, currentxpos) ** 2 +
                                    self._diff_pos(goalypos, currentypos) ** 2)
        previousdistance = math.sqrt(self._diff_pos(goalxpos, heroxpos) ** 2 +
                                     self._diff_pos(goalypos, heroypos) ** 2)
        if currentdistance < previousdistance:
            return True
        else:
            return False

    def _leave_throne_room(self):
        """Return boolean for leaving throne room for first time"""
        if self._current_map_id() == 4:
            self._escape_throne_room_lose_on_reset = True
        else:
            pass

    # %%
    # Series of definitions to determine agent penalty and reward

    def _death_penalty(self):
        """Return the reward for dying"""
        _reward = 0
        if self._is_dead():
            _reward += -1
        else:
            pass
        return _reward

    def _save_reward(self):
        """Reward the hero for saving significant progress"""
        _reward = 0
        if self._did_hero_just_save():
            _reward += -1  # always get 1 reward for exiting throne room, need at least two goals accomplished
            for classproperty in list(vars(DragonWarriorEnv).keys()):  # locate all properties of env
                if 'lose_on_reset' in classproperty:  # only select properties with values that lose on reset
                    if classproperty == True:  # if value will be lost on reset
                        if classproperty[:-14] == False:  # if value is not permanent
                            setattr(self, classproperty[:-14], True)  # store information on class property
                            setattr(self, classproperty, False)  # reset property reset value
                            _reward += 1  # reward for updating significant progress
        return _reward

    def _exp_reward(self):
        # return the reward based on party experience gained
        _reward = (self._current_exp() - self._hero_exp)
        self._hero_exp = self._current_exp()
        return _reward

    def _gold_reward(self):
        """Return the reward based on gold acquired"""
        if self._current_gold() > self._hero_gold:
            if self._current_map_id() == 5:
                self._pick_up_throne_room_gold_lose_on_reset = True  # only triggers the first time gold is picked up
        _reward = (self._current_gold() - self._hero_gold) / (self._hero_gold + 0.0001) * 100
        self._hero_gold = self._current_gold()
        return _reward

    def _hp_increase_reward(self):
        """Return the reward for recovering hit points"""
        _reward = 0
        if self._current_hp() > self._hero_hp:
            _reward += self._current_hp() - self._hero_hp
        else:
            pass
        self._hero_hp = self._current_hp()
        return _reward

    def _herb_reward(self):
        """Return the reward for gaining herbs"""
        _reward = 0
        # gaining herbs is good
        if self._current_magic_herbs() > self._hero_herb_count:
            _reward += 1
        # losing herbs
        if self._current_magic_herbs() < self._hero_herb_count:
            # good if you recover hp, but already accounted for in _hp_increase_reward
            if self._current_hp() > self._hero_hp:
                pass
            # bad if you do not recover hp
            else:
                _reward += -10
        # no change is neutral
        else:
            pass
        self._hero_herb_count = self._current_magic_herbs()
        return _reward

    # todo refine this for dungeon use
    def _torch_reward(self):
        """Return the reward for gaining torches"""
        _reward = 0
        if self._current_torches() > self._hero_torch_count:
            _reward += 1
            self._pick_up_throne_room_torch_lose_on_reset = True  # only matters for each new game
        else:
            pass
        self._hero_torch_count = self._current_torches()
        return _reward

    def _gain_magic_key_reward(self):
        """Return the reward for acquiring a magic key"""
        _reward = 0
        if self._current_magic_keys() > self._hero_magic_keys:
            _reward += 1
            self._pick_up_throne_room_key_lose_on_reset = True  # only matters for each new game
        else:
            pass
        # Update key value
        self._hero_magic_keys = self._current_magic_keys()
        return _reward

    def _gain_fairy_flute_reward(self):
        """Return the reward for acquiring the fairy flute"""
        _reward = 0
        if self._is_fairy_flute_in_inventory():
            if not self._obtain_fairy_flute:
                _reward += 15
                self._obtain_fairy_flute = True
            else:
                pass
        else:
            pass
        return _reward

    def _open_throne_door_reward(self):
        """Return the reward for opening a door using a magic key in the throne room.
        The only way to get rid of a key is to use it."""
        _reward = 0
        if self._current_map_id() == 5:  # only applies to keys used in throne room
            if self._current_magic_keys() < self._hero_magic_keys:
                _reward += 1
            else:
                pass
        else:
            pass
        return _reward

    # primitive distance based reward based on obtaining throne room key
    def _move_to_throne_room_key_reward(self):
        """Return the reward for moving closer to the throne room key chest"""
        _reward = 0
        if not self._pick_up_throne_room_key_lose_on_reset:
            if self._is_hero_closer_to_goal(self._throne_key_chest_x_pos, self._throne_key_chest_y_pos):
                try:
                    _reward += 0.01 / self._distance_to_goal(self._throne_key_chest_x_pos, self._throne_key_chest_y_pos)
                except:
                    _reward += 0.01  # should only trigger when distance_to_goal is zero
            if self._distance_to_goal(self._throne_key_chest_x_pos, self._throne_key_chest_y_pos) == 0:
                _reward += 0.02
            else:
                pass
        else:
            pass
        return _reward

    # primitive distance based reward based on obtaining throne room key
    def _exit_throne_room_reward(self):
        """Return the reward for moving closer to the throne room key chest"""
        _reward = 0
        if self._pick_up_throne_room_key_lose_on_reset:
            if self._hero_magic_keys > 0:
                if self._is_hero_closer_to_goal(self._throne_room_door_x_pos, self._throne_room_door_y_pos):
                    try:
                        _reward += 0.01 / self._distance_to_goal(self._throne_room_door_x_pos,
                                                                 self._throne_room_door_y_pos)
                    except:
                        _reward += 0.01
                if self._distance_to_goal(self._throne_room_door_x_pos, self._throne_room_door_y_pos) == 0:
                    _reward += 0.02
                else:
                    pass
            else:
                if self._is_hero_closer_to_goal(self._throne_room_stairs_xpos, self._throne_room_stairs_ypos):
                    try:
                        _reward += 0.01 / self._distance_to_goal(self._throne_room_stairs_xpos,
                                                                 self._throne_room_stairs_ypos)
                    except:
                        _reward += 0.01
                if self._distance_to_goal(self._throne_room_stairs_xpos, self._throne_room_stairs_ypos) == 0:
                    _reward += 0.02
                else:
                    pass
        else:
            pass
        return _reward

    # todo fix bad distance calculations based on map id, x/y pos
    def _move_to_fairy_flute_reward(self):
        """Return the reward for moving closer to the fairy flute"""
        _reward = 0
        if self._pick_up_throne_room_key and self._escape_throne_room and not self._obtain_fairy_flute:
            if self._current_map_id() == 5:
                if self._is_hero_closer_to_goal(self._throne_room_stairs_xpos, self._throne_room_stairs_ypos):
                    try:
                        _reward += 0.01 / self._distance_to_goal(self._throne_room_stairs_xpos,
                                                                 self._throne_room_stairs_ypos)
                    except:
                        _reward += 0.01
                else:
                    pass
            if self._current_map_id() == 4:
                if self._is_hero_closer_to_goal(10, 30):  # one tile of south Tantagel castle exit
                    try:
                        _reward += 0.01 / self._distance_to_goal(10, 30)
                    except:
                        _reward += 0.01
                else:
                    pass
            if self._current_map_id() == 1:
                if self._is_hero_closer_to_goal(104, 10):  # village of Kol on worldmap
                    try:
                        _reward += 0.01 / self._distance_to_goal(104, 10)
                    except:
                        _reward += 0.01
                else:
                    pass
            if self._current_map_id() == 7:
                if self._is_hero_closer_to_goal(9, 6):  # fairy flute location in Kol
                    try:
                        _reward += 0.01 / self._distance_to_goal(9, 6)
                    except:
                        _reward += 0.01
                else:
                    pass
            else:
                pass
        else:
            pass
        return _reward

    # first attempt to encourage exploration
    @property
    def _stationary_penalty(self):
        """Return the penalty for remaining in the same x/y pos and map id"""
        _reward = 0
        if self._is_same_pos():
            # significant penalty for trying comboactions. If comboaction leads to a positive reward, the positive
            # reward will outweigh the penalty. If the comboaction does not lead to a positive reward, the larger
            # penalty provides massive incentive to not use comboactions that have no effect
            if DragonWarriorEnv.herocurrentcomboaction in self._dragon_warrior_combo_actions:
                if self.is_command_window_open():  # combo actions press A, which is better than directions
                    _reward += -1e-4
                else:
                    _reward += -1e-4
            # small penalty for standing still with no action
            else:
                if self.is_command_window_open():
                    _reward += -1e-4
                else:
                    _reward += -1e-4
        else:
            pass
        self._hero_x_pos = self._current_map_x_pos()
        self._hero_y_pos = self._current_map_y_pos()
        self._hero_map_id = self._current_map_id()
        return _reward

    # todo aggregate all current -> hero updates into a single function, run after reward function
    def _get_reward(self):
        """Return the reward after a step occurs"""
        self._frame_buffering()
        # print(self._exp_reward() , self._open_throne_door_reward() , self._gain_magic_key_reward() ,
        #         self._herb_reward() , self._gold_reward() ,
        #         self._torch_reward() , self._hp_increase_reward() , self._move_to_throne_room_key_reward() ,
        #         self._exit_throne_room_reward() , self._move_to_fairy_flute_reward() ,
        #         self._gain_fairy_flute_reward() , self._save_reward() ,
        #         self._stationary_penalty , self._death_penalty())
        return (self._exp_reward() + self._open_throne_door_reward() +
                self._herb_reward() + self._gold_reward() +
                self._torch_reward() + self._hp_increase_reward() + self._move_to_throne_room_key_reward() +
                self._exit_throne_room_reward() + self._move_to_fairy_flute_reward() +
                self._gain_fairy_flute_reward() + self._save_reward() +
                self._stationary_penalty + self._gain_magic_key_reward() + self._death_penalty())

    def _get_done(self):
        """Return True if the episode is over, False otherwise"""
        return False

    # include RAM info needed for agent (map_id, hero_stats, map_position, etc)
    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            throne_room_key=self._pick_up_throne_room_key_lose_on_reset,
            throne_room_gold=self._pick_up_throne_room_gold_lose_on_reset,
            throne_room_torch=self._pick_up_throne_room_torch_lose_on_reset,
            exit_throne_room=self._escape_throne_room_lose_on_reset,
            exit_throne_room_perm = self._escape_throne_room,
            found_fairy_flute=self._obtain_fairy_flute,
            hero_exp=self._hero_exp,
            hero_gold=self._hero_gold,
            hero_map_id=self._hero_map_id,
            hero_xpos=self._hero_x_pos,
            hero_ypos=self._hero_y_pos,
            hero_weapon=self._current_weapon(),
            hero_armor=self._current_armor(),
            hero_shield=self._current_shield(),
        )


# explicitly define the outward facing API for the module
__all__ = [DragonWarriorEnv.__name__]
