# DragonWarriorRL

## Goal
The project uses **reinforcement learning, Deep Q learning, and neural networks** to play the video game _Dragon Warrior_ for the Nintendo Entertainment System (NES).

## Method
The **Q learning model** (agent) interacts with an environment (Python scripted NES) to determine optimal outcomes based on a custom reward function. I currently employ the following techniques:
- 2 dimensional convulation neural networks to interpret the NES screen as input parameters into the Q function
- Simple action space (up, down, left, right, A, B, start)
- **Adam optimization** of neural network. Stochastic gradient descent method based on adaptive estimation of first-order and second-order moments.

## Challenges
1. The game has two sub-environments: the map, and combat. 
  - Combat has four options: fight, item, magic, run. Item and Magic both have sub-menu options (complicated), while fight automatically attacks the only target available (simple) and run flees (simple).
  - Map navigation brings up a sub-menu of 8 choices whenever the A button is pressed (Search, Talk, Stairs, Item, Magic, etc).

## Outcome
This is still a work in progress, but feel free to let me know if you have any thoughts or ideas!
