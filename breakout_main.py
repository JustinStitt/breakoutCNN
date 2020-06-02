import sys, os, time
import gym
import numpy as np
from agent import DDQNAgent
from utils import plot_learning_curve, make_env

render = False
to_save = True
to_load = False

print('Render: ', render)
print('Save: ', to_save)
print('Load: ', to_load)

if __name__ == '__main__':
    env = make_env('BreakoutNoFrameskip-v4')#from utils

    best_score = -np.inf
    agent = DDQNAgent(gamma = 0.99, epsilon = 1.0, lr = 0.0001789,
                        input_dims = (env.observation_space.shape),
                        n_actions = env.action_space.n, mem_size = 60_000,
                        eps_min = 0.1, batch_size = 32, replace = 7_500, eps_dec = 1e-5)
    #try replace = 10_000 and lr = 0.000091
    if os.path.isfile('trained_model/local.pt') and to_load:
        agent.load_agent()
        agent.epsilon = agent.eps_min#we currently aren't saving epsilon so just set to min

    epochs = 1_000

    figure_file = 'plots/post_training_plot.png'
    steps = 0
    scores, eps_history, steps_array = [], [] ,[]

    start_time = time.time()
    time_elapsed = time.time()
    end_time = start_time

    for i in range(epochs):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            if render:
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            agent.store_memory(observation, action, reward, observation_, int(done))
            agent.learn()

            observation = observation_
            steps += 1
        scores.append(score)
        steps_array.append(steps)

        avg_score = np.mean(scores[-100:])#avg of last 100 scores
        passed_time = time.time() - time_elapsed
        time_elapsed = time.time()
        print('epoch: ', i, ' score: ', score, ' avg score: %.2f' % avg_score,
                    ' best score: %.2f' % best_score, ' epsilon: %.3f' % agent.epsilon,
                        ' steps: ', steps, ' time elapsed: %.3f' % passed_time)
        pass
        if i % 50 == 0 and i > 0 and to_save:
            agent.save_agent()

        if avg_score > best_score:
            best_score = avg_score

        eps_history.append(agent.epsilon)
    total_time = time.time() - start_time
    print('total time elapsed (seconds) : %.3f' % total_time)

    if to_save:
        agent.save_agent()

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
