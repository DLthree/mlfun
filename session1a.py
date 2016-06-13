import gym
import numpy as np
from qlearner import QLearner

paddle_line = 192
chris_sux = 189
tim_sux = 40

def find_paddle(state):
    line = state[paddle_line,8:-8,0]
    indices = np.where(line == 200)
    return np.mean(indices)

def find_ball(a,b):
    diff = b-a
    diff = diff[tim_sux:chris_sux,:,0]
    indices = np.where(diff == 200)
    y = np.mean(indices[0]) + tim_sux
    x = np.mean(indices[1]) # chris_sux
    return (x,y)

env = gym.make('Breakout-v0')
learner = QLearner(num_states=500, num_actions=env.action_space.n)
for i_episode in range(2000):
    observation = env.reset()
    action = learner.set_initial_state(0)
    prev = observation
    total_reward = 0
    for t in range(10000):
        # env.render()
        prev = observation
        observation, reward, done, info = env.step(action)
        total_reward += reward
        paddle = find_paddle(observation)
        x,y = find_ball(prev, observation)
        try:
            feature = int(paddle - x)
            action = learner.move(feature, reward)
        except ValueError:
            feature = 250
            action = learner.move(feature, reward, force_random=True)

        if done:
            print("Episode finished after {} timesteps.  {} reward".format(t+1, total_reward))
            break
