import random
import gym_simplegrid
import gymnasium as gym
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils.save_video import save_video

FOLDER_NAME = "out_old"
os.makedirs(f"log/{FOLDER_NAME}")

learning_rate = 0.7
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.005

max_episodes = 500
test_episodes = 100
max_steps = 30

obstacle_map = [
    "00001000",
    "10010000",
    "00000001",
    "01000001",
]

env = gym.make(
    'SimpleGrid-v0',
    obstacle_map=obstacle_map,
    render_mode='rgb_array_list',
)


class DelayedReward(gym.Wrapper):
    def __init__(self, env, T, high_discount, low_discount):
        gym.Wrapper.__init__(self, env)
        self.T = T
        self.high_discount = high_discount
        self.low_discount = low_discount
        self.timestep = 0

    def step(self, action):
        ob, reward, done, _, info = self.env.step(action)
        #reward *= self.discount()
        self.timestep += 1
        return ob, reward, done, _, info, self.timestep

    def discount(self):
        if self.timestep < self.T:
            return self.high_discount
        else:
            return self.low_discount


#env = DelayedReward(env, 6, 1.5, 0.5)
obs, info = env.reset(options={'start_loc': 0, 'goal_loc': 24})
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

Q = np.zeros((env.observation_space.n, env.action_space.n))
training_rewards = []
epsilons = []


for episode in range(max_episodes):
    print(episode)
    total_training_rewards = 0
    state = env.reset(options={'start_loc': 0, 'goal_loc': 24})
    state = state[0]

    for step in range(max_steps):
        # Choosing an action given the states based on a random number
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state, :])
        else:
            action = env.action_space.sample()

        ### STEPs 3 & 4: performing the action and getting the reward
        # Taking the action and getting the reward and outcome state
        obs, reward, done, _, info = env.step(action)

        ### STEP 5: update the Q-table
        # Updating the Q-table using the Bellman equation
        Q[state, action] = Q[state, action] + learning_rate * (
                reward + np.max(Q[obs, :]) - Q[state, action])
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

"""for t in range(500):
    # img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")

    action = env.action_space.sample()
    f.write(f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n")

    if done:
        break

    obs, rew, done, _, info, _ = env.step(action)"""

"""if env.render_mode == 'rgb_array_list':
    frames = env.render()
    save_video(frames, f"log/{FOLDER_NAME}", fps=env.fps)"""

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
f.write(f"test episodes: {test_episodes}\n")
f.write(f"max steps: {max_steps}\n")
f.close()

env.close()
