import random
from typing import Any

import gym_simplegrid
import gymnasium as gym
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperObsType
from gymnasium.utils.save_video import save_video

start_time = dt.now()

FOLDER_NAME = "out_example"
os.makedirs(f"log/{FOLDER_NAME}")

learning_rate = 0.78
epsilon = 1
max_epsilon = 1
min_epsilon = 0.05
decay = 0.002

max_episodes = 500
rand_episodes = 0
max_steps = 40
min_steps = 40

goal_step = 10
high_discount = 1.5
low_discount = 0.8


obstacle_map = [
    "0000010",
    "1000000",
    "0000000",
]

"""available_starts = []
for i in range(7*4):
    available_starts.append(i)
available_starts.remove(5)
available_starts.remove(6)
available_starts.remove(7)
available_starts.remove(26)
available_starts.remove(22)"""


env = gym.make(
    'SimpleGrid-v0',
    obstacle_map=obstacle_map,
    render_mode='rgb_array_list',
)


#class NewSimple(Env):


class DelayedReward(gym.Wrapper):
    MOVES: dict[int, tuple] = {
        0: (-1, 0),  # UP
        1: (1, 0),  # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1)#,  # RIGHT
        #4: (0, 0)  # STAY
    }

    def __init__(self, env, T, high_discount, low_discount):
        gym.Wrapper.__init__(self, env)
        #self.env.n_iter = 0
        #self.unwrapped.MOVES = self.MOVES
        self.T = T
        self.high_discount = high_discount
        self.low_discount = low_discount
        self.timestep = 0

        #self.action_space = spaces.Discrete(len(self.MOVES))

    def temp_reward(self, rew):
        if 0.9 < rew < 1.1:
            return 1.0
        else:
            return rew

    def pseudo_reset(self, options):
        return self.env.reset(options=options)

    def step(self, action):
        ob, reward, done, _, info = self.env.step(action)
        reward = self.temp_reward(reward) * self.discount()
        self.timestep += 1
        return ob, reward, done, _, info, self.timestep

    def discount(self):
        if self.timestep < self.T:
            return self.high_discount
        else:
            return self.low_discount

    """def step(self, action: int):
        row, col = self.env.agent_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.env.reward = self.env.get_reward(target_row, target_col)

        # Check if the move is valid
        if self.env.is_in_bounds(target_row, target_col) and self.env.is_free(target_row, target_col):
            self.env.agent_xy = (target_row, target_col)
            self.env.done = self.env.on_goal()

        self.env.n_iter += 1

        #  if self.render_mode == "human":
        self.render()

        rew = self.env.reward * self.discount()

        return self.env.get_obs(), rew, self.env.done, False, self.env.get_info(), self.timestep"""




env = DelayedReward(env, goal_step, high_discount, low_discount)
"""obs, info ="""
env.reset(options={'start_loc': 0, 'goal_loc': 14})
rew = env.unwrapped.reward
done = env.unwrapped.done


"""FOLDER_NAME = "out3"
os.makedirs(f"log/{FOLDER_NAME}")
with open(f"log/{FOLDER_NAME}/history.csv", 'w') as f:
    f.write(f"step,x,y,reward,done,action\n")

    for t in range(500):
        # img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")

        action = env.action_space.sample()
        f.write(f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n")

        if done:
            break

        obs, rew, done, _, info, _ = env.step(action)"""

Q = np.zeros((env.observation_space.n, env.action_space.n, max_steps))
training_rewards = []
epsilons = []


for episode in range(max_episodes):
    print(episode)
    total_training_rewards = 0
    #state = env.reset(options={'start_loc': 0, 'goal_loc': 6})
    #state = state[0]

    curr_steps = int(max_steps - ((max_steps - min_steps) * ((episode - rand_episodes) / (max_episodes - rand_episodes))))

    if episode < rand_episodes:
        curr_steps = max_steps
        state = env.reset(options={'start_loc': 0, 'goal_loc': 14})
        state = state[0]

        for step in range(curr_steps - 1):
            action = env.action_space.sample()
            obs, reward, done, _, info, _ = env.step(action)

            Q[state, action, step] = Q[state, action, step] + learning_rate * (
                    reward + np.max(Q[obs, :, step + 1]) - Q[state, action, step])

            total_training_rewards += reward
            state = obs
    else:
        state = env.reset(options={'start_loc': 0, 'goal_loc': 14})
        state = state[0]
        for step in range(curr_steps - 1):
        # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, : , step])
            else:
                action = env.action_space.sample()

            ### STEPs 3 & 4: performing the action and getting the reward
            # Taking the action and getting the reward and outcome state
            obs, reward, done, _, info, _ = env.step(action)

            ### STEP 5: update the Q-table
            # Updating the Q-table using the Bellman equation
            Q[state, action, step] = Q[state, action, step] + learning_rate * (
                    reward + np.max(Q[obs, : , step+1]) - Q[state, action, step])
            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = obs

            # Ending the episode
            if done:
                break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    # Adding the total reward and reduced epsilon values
    training_rewards.append(total_training_rewards)
    epsilons.append(epsilon)
    if episode == max_episodes-1:
        if env.render_mode == 'rgb_array_list':
            frames = env.render()
            save_video(frames, f"log/{FOLDER_NAME}", fps=env.fps)


print(f"done in {step} steps")
end_time = dt.now()

total_time = end_time - start_time
print(f"\n{total_time.total_seconds()}")

x = range(max_episodes)
fig = plt.plot(x, training_rewards)
plt.xlabel('Episode')
plt.ylabel('Training total reward')
plt.title('Total rewards over all episodes in training')

plt.savefig(f"./log/{FOLDER_NAME}/fig.png")
plt.close()

f = open(f"./log/{FOLDER_NAME}/stats.txt", "w")
f.write(f"learning rate: {learning_rate}\n")
f.write(f"training episodes: {max_episodes}\n")
f.write(f"test episodes: {rand_episodes}\n")
f.write(f"max steps: {max_steps}\n")
f.write(f"goal steps: {goal_step}\n")
f.write(f"high discount: {high_discount}\n")
f.write(f"low discount: {low_discount}\n")
f.write(f"runtime: {total_time.total_seconds()}\n")
f.close()

env.close()
