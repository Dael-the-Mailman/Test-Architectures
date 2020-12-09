import numpy as np
import gym
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from agent import Agent
from DQNet import DQNet
from DuelDQNet import DuelDQNet
from DuelDDQNet import DuelDDQNet
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    nets = [DQNet, DuelDQNet, DuelDDQNet]
    print('Choose network from the following selection:')
    for index, option in enumerate(nets):
        print(f' {index}: {option.name}')
    net = nets[int(input())]
    
    agent = Agent(net, gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
            eps_end=0.01, input_dims=[8], lr=0.003)
    
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)