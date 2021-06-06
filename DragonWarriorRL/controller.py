import utilities


class HumanPresser:
    '''Provides a set of methods that allows a human to control the emulator and to update the model and Q-table as if
    the bot made the choices of actions. The method names picked for brevity to make it less verbose to use in an
    interactive interpreter.'''

    def __init__(self, actor):
        self.actor = actor

    def w(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['up'])
        self.actor.env.render()

    def a(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['left'])
        self.actor.env.render()

    def s(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['down'])
        self.actor.env.render()

    def d(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['right'])
        self.actor.env.render()

    def stairs(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['menucol0row2'])
        self.actor.env.render()

    def door(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['menucol1row2'])
        self.actor.env.render()

    def take(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['menucol1row3'])
        self.actor.env.render()

    def attack(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['menucol0row0'])
        self.actor.env.render()

    def b(self, laststepwassuccessful=True, printtiming=True):
        # if we don't pass anything assume the action when through and update the space.
        if self.actor.results is not None and laststepwassuccessful:
            self.actor.dopostprocessingformpreviousstep()
        self.actor.results = self.actor.doaction(self.actor.env.dict_comboactionsindextoname['B'])
        self.actor.env.render()

# %%

if __name__ == '__main__':

    # The actor should already be set up using at least parts of run_dragon_warrior.py
    s = HumanPresser(actor=actor)

    # %%

    env.pressbutton('A')
    env.pressbutton('left')
    env.pressbutton('right')
    env.pressbutton('up')
    env.pressbutton('down')
    env.pressbutton('B')

    # %%

    s.d()
    s.take()
    s.d()
    s.take()
    s.d()
    [s.d() for index in range(3)]
    s.w()
    [s.w() for index in range(3)]
    s.a()
    s.a()
    s.take()
    s.d()
    s.d()
    s.s()
    [s.s() for index in range(5)]
    s.a()
    [s.a() for index in range(5)]
    s.s()
    s.door()
    s.s()
    s.s()
    s.d()
    [s.d() for index in range(5)]
    s.stairs()
    s.d()
    s.d()
    s.s()
    [s.s() for index in range(8)]
    s.a()
    [s.s() for index in range(6)]
