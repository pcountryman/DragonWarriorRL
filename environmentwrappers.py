import gym
from gym import Env
from gym import Wrapper

class ButtonRemapper(Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        'right': 0b10000000,
        'left': 0b01000000,
        'down': 0b00100000,
        'up': 0b00010000,
        'start': 0b00001000,
        'select': 0b00000100,
        'B': 0b00000010,
        'A': 0b00000001,
        'NOOP': 0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: Env, list_controlleractions: list, list_comboactions: list, renderflag=False):
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            list_controlleractions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super().__init__(env)
        self.renderflag = renderflag
        self.list_controlleractions = list_controlleractions
        self.list_comboactions = list_comboactions
        self.dict_comboactionsindextoname = dict()

        for index, comboactionname in enumerate(list_comboactions):
            self.dict_comboactionsindextoname[comboactionname[0]] = index

        self.env = env
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(list_controlleractions))
        self.action_space_combo = gym.spaces.Discrete(len(list_comboactions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(list_controlleractions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = ' '.join(button_list)

        self.dict_combobuttonpresses = {'left': ['left'], 'right': ['right'], 'up': ['up'], 'down': ['down'],
                                        'menucol0row0': ['A', 'B', 'B'],
                                        'menucol0row1': ['A', 'down', 'A', 'B'],
                                        'menucol0row2': ['A', 'down', 'down', 'A', 'B'],
                                        'menucol0row3': ['A', 'down', 'down', 'down', 'A', 'B'],
                                        'menucol1row0': ['A', 'right', 'A', 'B'],
                                        'menucol1row1': ['A', 'right', 'down', 'A', 'B'],
                                        'menucol1row2': ['A', 'right', 'down', 'down', 'A', 'B'],
                                        'menucol1row3': ['A', 'right', 'down', 'down', 'down', 'A', 'B'],
                                        'A': ['A'], 'B': ['B']
                                        }

        self.dict_takesactionnamereturnsbuttonindex = dict()
        # env._action_meanings = {0: 'NOOP', 1: 'right', 2: 'left', 3: 'up', 4: 'down', 5: 'A', 6: 'B'}
        for actionindex in self._action_meanings.keys():
            actionname = self._action_meanings[actionindex]
            self.dict_takesactionnamereturnsbuttonindex[actionname] = actionindex

    def step(self, action):
        """
        Take a step using the given action.

        Args:
            action (int): the discrete action to perform

        Returns:
            a tuple of:
            - (numpy.ndarray) the state as a result of the action
            - (float) the reward achieved by taking the action
            - (bool) a flag denoting whether the episode has ended
            - (dict) a dictionary of extra information

        """
        # Old version:
        # take the step and record the output
        # return self.env.step(self._action_map[action])

        # Where [['left'], ['right'], ['up'],['down']] are the combo button pressing names. If it is these things
        # we need extra presses to increase the probability that the emulator processes the action press.
        doextrapress = self.dict_combobuttonpresses[self.list_comboactions[action][0]] in [['left'], ['right'], ['up'],
                                                                                           ['down']]

        # without this, some rewards obtained during combo actions executions were missed by the system
        reward_for_combo_action = 0

        # Iterate through the simple button presses that the combo button presses are built from and convert them
        # into the corresponding bits that the emulator (env.step) is expecting.
        for actionname in self.dict_combobuttonpresses[self.list_comboactions[action][0]]:

            # Recall again we only do extra presses if actionname in [['left'], ['right'], ['up'],['down']]
            if doextrapress:
                # TODO: See if we can come up with more effective logic. Also implement frame buffering.
                self.pressbutton(actionname)

            # this will update multiple times, but we will only return the last value.
            next_state, reward, done, info = self.env.step(
                self._action_map[self.dict_takesactionnamereturnsbuttonindex[actionname]])
            self.env.frame_advance(0)
            reward_for_combo_action += reward

        return next_state, reward_for_combo_action, done, info

    def reset(self):
        """Reset the environment and return the initial observation."""
        return self.env.reset()

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        # get the old mapping of keys to actions
        old_keys_to_action = self.env.unwrapped.get_keys_to_action()
        # invert the keys to action mapping to lookup key combos by action
        action_to_keys = {v: k for k, v in old_keys_to_action.items()}
        # create a new mapping of keys to actions
        keys_to_action = {}
        # iterate over the actions and their byte values in this mapper
        for action, byte in self._action_map.items():
            # get the keys to press for the action
            keys = action_to_keys[byte]
            # set the keys value in the dictionary to the current discrete act
            keys_to_action[keys] = action

        return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        list_controlleractions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in list_controlleractions]

    def presscombobutton(self, combobuttonname):
        for buttonname in self.dict_combobuttonpresses[combobuttonname]:
            self.pressbutton(buttonname)

    def advanceframes(self, frames=100):
        for attempt in range(frames):
            self.env.frame_advance(0)
            if self.renderflag:
                self.env.render()

    # TODO: update button press to use RAM information to wait for actions to be finished instead of lookup tables.
    def pressbutton(self, button, trailingnoons=None, presses=None):
        '''This executes combo button presses mean to make it more likely for the emulator to accept the button press.'''

        # Menu does not need a bunch of extra presses.
        # TODO: make sure the default presses is still good when we implement frame buffering.
        if presses is None:
            # If the menu is open.
            if self.env.is_command_window_open():
                presses = 1
            else:
                presses = 20

        dict_trailingnoons = {
            'NOOP': 0,
            'right': 100,
            'left': 100,
            'up': 100,
            'down': 100,
            'A': 250,
            'B': 25
        }


        for press in range(presses):
            self.frame_advance(self._button_map[button])
            if self.renderflag:
                self.render()

        # if trailingnoons is None:
        #     trailingnoons = dict_trailingnoons[button]
        #
        # for index in range(trailingnoons):
        #     self.env.frame_advance(self._button_map['NOOP'])
        #     if self.renderflag:
        #         self.env.render()