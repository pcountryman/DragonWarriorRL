import gym
from gym import Env
from gym import Wrapper

class ButtonRemapper(Wrapper):


    def __init__(self, env: Env, actions, renderflag=False):
        super().__init__(env)
        # # create the new action space
        # self.action_space = gym.spaces.Discrete(len(actions))
        # # create the action map from the list of discrete actions
        # self._action_map = {}
        # self._action_meanings = {}
        # # iterate over all the actions (as button lists)
        # for action, button_list in enumerate(actions):
        #     # the value of this action's bitmap
        #     byte_action = 0
        #     # iterate over the buttons in this button list
        #     for button in button_list:
        #         byte_action |= self._button_map[button]
        #     # set this action maps value to the byte action value
        #     self._action_map[action] = byte_action
        #     self._action_meanings[action] = ' '.join(button_list)

        self.dict_combobuttonpresses = {'left': ['left'], 'right': ['right'], 'up': ['up'], 'down': ['down'],
                                   'take': ['A', 'right', 'down', 'down', 'down', 'A', 'A'],
                                   'door': ['A', 'right', 'down', 'down', 'A', 'A'],
                                   'stairs': ['A', 'down', 'down', 'A', 'A'],
                                   'A': ['A'], 'B': ['B']
                                   }
        self.env = env
        # self.action_space = gym.spaces.Discrete(len(actions))
        self.renderflag = renderflag
        self.actions = actions

        self.dict_actionnametoindexes = dict()
        for actionindex in env._action_meanings.keys():
            actionname = env._action_meanings[actionindex]
            self.dict_actionnametoindexes[actionname] = actionindex


    def presscombobutton(self, combobuttonname):
        for buttonname in self.dict_combobuttonpresses[combobuttonname]:
            self.pressbutton(buttonname)

    def advanceframes(self, frames=100):

        for attempt in range(frames):
            self.env.frame_advance(0)
            if self.renderflag:
                self.env.render()

    def pressbutton(self, button, trailingnoons=None, presses=None):
        if presses is None:
            if self.env.command_window_state():
                presses = 1
            else:
                presses = 11

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
            self.env.frame_advance(self.env._button_map[button])
            if self.renderflag:
                self.env.render()
        if trailingnoons is None:
            trailingnoons = dict_trailingnoons[button]

        for index in range(trailingnoons):
            self.env.frame_advance(self.env._button_map['NOOP'])
            if self.renderflag:
                self.env.render()

    def doaction(self, action, trailingnoons=None, presses=None):
        dict_inverse = dict()

        for key in self.env._button_map.keys():
            value = self.env._button_map[key]
            dict_inverse[value] = key

        if presses is None:
            if self.env.command_window_state():
                presses = 1
            else:
                presses = 11

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
            self.env.frame_advance(action)
            if self.renderflag:
                self.env.render()
        if trailingnoons is None:
            trailingnoons = dict_trailingnoons[dict_inverse[action]]

        for index in range(trailingnoons):
            self.env.frame_advance(self.env._button_map['NOOP'])
            if self.renderflag:
                self.env.render()

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
        # take the step and record the output
        for actionname in self.dict_combobuttonpresses[self.actions[action][0]]:
            # this will update multiple times, but we will only return the last value.
            next_state, reward, done, info = self.env.step(self.env._action_map[self.dict_actionnametoindexes[actionname]])
        return next_state, reward, done, info



# %%

