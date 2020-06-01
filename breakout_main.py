import gym
import sys, os, time
import numpy as np
from DQN_CNN import Agent
from matplotlib import pyplot
from preprocessing import preprocess as pp

render = False
to_load = False
to_save = True

if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')

    dims = pp(env.reset()).shape#take sample of environment and preprocess it to determine input's shape

    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 32,
                    input_dims = dims, n_actions = env.action_space.n,  lr = 0.0000625, eps_dec = 1e-5)
    #observation space.shape = (210,160,3)
    #pp observationspace.shape = (105,80,1)
    if os.path.isfile('saved_models/saved_model_local.pt') and to_load:
        agent.load_agent()
        agent.epsilon = agent.eps_min

    scores = []
    epochs = 10_000
    score = 0

    for i in range(epochs):
        if i % 1 == 0 and i > 0:
            avg_score = np.mean(scores[-100:])#avg of last 100 scores
            print('epoch: ', i, 'score: ', score, 'avg score: %.3f ' % avg_score,
                        'epsilon: %.4f' % agent.epsilon)

        if i % 50 == 0 and i > 0 and to_save:
            agent.save_agent()

        score = 0
        observation = env.reset()
        observation = pp(observation)
        #print('-==OB SHAPE==-', observation.shape)
        #new_ob = pp(observation)
        #print('-=PP OBS=-', new_ob.shape)


        done = False
        while not done:
            if render:
                env.render()
                #im_to_show = observation[::1,::1,-1]
                #print('shown image shape: ', im_to_show.shape)
                #pyplot.imshow(im_to_show, cmap='gray')
                #pyplot.show()
            action = agent.choose_action(observation)#adding to frames
            observation_, reward, done, info = env.step(action)
            observation_ = pp(observation_)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)#adding to frames
            agent.learn()
            observation = observation_
        scores.append(score)
    if to_save:
        agent.save_agent()
