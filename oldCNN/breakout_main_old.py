#import argparse, os
import gym
import os, sys
import time
import numpy as np
from DQN_CNN import DQNAgent

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    #print('-==ENV SPACE==-', env.observation_space)
    agent = DQNAgent(gamma = 0.99, epsilon = 1.0, lr = 1e-4,
                    n_actions = env.action_space.n, input_dims = env.observation_space.shape,
                    mem_size = 20000, batch_size = 32, eps_min = 0.1, eps_dec = 1e-5)#.shape for input_dims

    scores = []
    eps_history = []
    n_games = 625
    score = 0
    n_steps = 0
    steps_array = []
    best_score = -np.inf
    load_cp = False

    if(load_cp and os.path.exists('checkpoints/cnn_model.pt')):
        print('...loaded checkpoint...')
        agent.load_models()

    for i in range(n_games):
        t0 = time.time()
        done = False
        observation = env.reset()
        score = 0
        while not done:
            #env.render()# to draw

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation,action,reward,observation_,int(done))
            agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])#avg of last 100 scores
        t_d = time.time() - t0
        print('episode ', i, ' score ', score, ' average score %.2f' % avg_score, ' best score %.2f ' % best_score,
            ' epsilon %.2f ' % agent.epsilon, ' steps ', n_steps, ' time ', t_d)
        if score > best_score:
            best_score = score

        eps_history.append(agent.epsilon)
        x = [i + 1 for i in range(len(scores))]
        #plot_learning_curve(steps_array, scores, eps_history, figure_file)
    #all games are done. lets save a checkpoint
    agent.q_eval.save_checkpoint()
