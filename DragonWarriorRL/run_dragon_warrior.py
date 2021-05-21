from dragon_warrior_env import DragonWarriorEnv
from actions import dragon_warrior_actions
from nes_py.wrappers import JoypadSpace
from dqn_agent import DQNAgent
import time


# todo add DQN agent!

env = DragonWarriorEnv()
env = JoypadSpace(env, dragon_warrior_actions)
env.reset()

# Parameters
states = (240, 256, 3)
actions = env.action_space.n

agent = DQNAgent(states=states, actions=actions, max_memory=1000000, double_q=True)

# Timing
start = time.time()
step = 0

state = env.reset()

total_reward = 0
iter = 0

while True:
# for _ in range(160000):
    env.render()
    # run agent
    action = agent.run(state=state)
    #perform action
    next_state, reward, done, info = env.step(action=action)
    #remember transition
    agent.add(experience=(state, next_state, action, reward, done))

    # update agent
    agent.learn()

    # total reward
    total_reward += reward

    # update state
    state = next_state

    # increment
    iter += 1


# for _ in range(50000):
#     env.render()
#     state, reward, done, info = env.step(env.action_space.sample())
#     # env.step(env.action_space.sample())
# env.close()

# env = NESEnv
# # action space is large due to number of buttons and button combinations. Need to figure out which numbers
# # in action space correspond to which button presses. FF doesn't need button combinations, just directions
# # A, B, start, select (8). Eventually also soft and hard reset
# print(env.action_space)
# # observation space is pixel based and RGB color coding of the pixels
# print(env.observation_space)
# # print(env.observation_space.high)
# # print(env.observation_space.low)