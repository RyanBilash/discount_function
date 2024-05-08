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

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = dt.now()

FOLDER_NAME = "out_gpu15"
os.makedirs(f"log/{FOLDER_NAME}")

learning_rate = 0.6
epsilon = 1
max_epsilon = 1
min_epsilon = 0.15
decay = 0.002

max_episodes = 900
rand_episodes = 4
max_steps = 50
min_steps = 50

goal_step = 10
high_discount = 1.8
low_discount = 0.9


obstacle_map = [
    "00010010000",
    "00010000000",
    "00000000000",
    "01000010000",
    "01000010000",
]
available_starts = []
for i in range(11*5):
    available_starts.append(i)
available_starts.remove(27)
available_starts.remove(3)
available_starts.remove(6)
available_starts.remove(14)
available_starts.remove(34)
available_starts.remove(39)
available_starts.remove(45)
available_starts.remove(50)

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
        3: (0, 1),  # RIGHT
        4: (0, 0)  # STAY
    }

    def __init__(self, env, T, high_discount, low_discount):
        gym.Wrapper.__init__(self, env)
        self.env.n_iter = 0
        self.unwrapped.MOVES = self.MOVES
        self.T = T
        self.high_discount = high_discount
        self.low_discount = low_discount
        self.timestep = 0

        self.action_space = spaces.Discrete(len(self.MOVES))

    def pseudo_reset(self, options):
        return self.env.reset(options=options)

    def temp_reward(self, rew):
        return rew
        if 0.9 < rew < 1.1:
            return 1.0
        else:
            return rew

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




env = DelayedReward(env, goal_step, high_discount, low_discount)
"""obs, info ="""
env.reset(options={'start_loc': 0, 'goal_loc': 27})
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

Q = torch.from_numpy(np.zeros((env.observation_space.n, env.action_space.n, max_steps)))
Q = Q.to(device=device)
training_rewards = []
epsilons = []


for episode in range(max_episodes):
    print(episode)
    total_training_rewards = 0

    if episode < rand_episodes:
        curr_steps = max_steps
        state = env.reset(options={'start_loc': random.choice(available_starts), 'goal_loc': 27})
        state = state[0]

        for step in range(curr_steps - 1):
            if episode < rand_episodes:
                action = env.action_space.sample()
                obs, reward, done, _, info, _ = env.step(action)

                Q[state, action, step] = Q[state, action, step] + learning_rate * (
                        reward + torch.max(Q[obs, :, step + 1]) - Q[state, action, step])

                total_training_rewards += reward
                state = obs
    else:
        curr_steps = int(
            max_steps - ((max_steps - min_steps) * ((episode - rand_episodes) / (max_episodes - rand_episodes))))

        state = env.reset(options={'start_loc': 0, 'goal_loc': 27})
        state = state[0]

        for step in range(curr_steps-1):
        # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = torch.argmax(Q[state, : , step]).item()
            else:
                action = env.action_space.sample()

            ### STEPs 3 & 4: performing the action and getting the reward
            # Taking the action and getting the reward and outcome state
            obs, reward, done, _, info, _ = env.step(action)

            ### STEP 5: update the Q-table
            # Updating the Q-table using the Bellman equation
            Q[state, action, step] = Q[state, action, step] + learning_rate * (
                    reward + torch.max(Q[obs, : , step+1]) - Q[state, action, step])
            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = obs

            # Ending the episode
            if done:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * (episode - rand_episodes))

    # Adding the total reward and reduced epsilon values
    training_rewards.append(total_training_rewards)
    epsilons.append(epsilon)


print(f"done in {step} steps")
if env.render_mode == 'rgb_array_list':
    frames = env.render()
    save_video(frames, f"log/{FOLDER_NAME}", fps=env.fps)

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
