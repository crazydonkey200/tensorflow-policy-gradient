import argparse
import cProfile as profile
import os
import sys
import time

import gym
import tensorflow as tf
import numpy as np
import policy_gradient as pg
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str,
                        default='CartPole-v0',
                        help='Environment name.')

    parser.add_argument('--save_path', type=str,
                        default='',
                        help='Path to save experiments.')

    parser.add_argument('--config_file', type=str,
                        default='',
                        help='Json file containing configurations.')

    parser.add_argument('--n_iters', type=int,
                        default=200,
                        help='Number of iterations.')

    parser.add_argument('--evaluate_freq', type=int,
                        default=10,
                        help='How often to evaluate on a game.')

    parser.add_argument('--max_steps', type=int,
                        default=200,
                        help='Upper limit of episode length.')

    
    # Parameters for unittesting the implementation.
    parser.add_argument('--record', dest='record', action='store_true',
                        help='Whether to record and save this experiment.')
    parser.set_defaults(record=False)

    args = parser.parse_args()

    if not args.save_path:
        args.save_path =  os.path.join('/tmp/', args.env + '-tmp-experiment')
        
    env = gym.make(args.env)

    if args.record:
        env.monitor.start(args.save_path, force=True)
    
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        agent = pg.NNAgent(env.action_space, env.observation_space,
                           max_steps=args.max_steps, **config)
    else:
        agent = pg.NNAgent(env.action_space, env.observation_space,
                           max_steps=args.max_steps,
                           learning_rate=100.0, discount=0.98,
                           use_softmax_bias=False,
                           use_rnn=False)

    n_iters = args.n_iters
    iter_num = range(n_iters)
    returns = []
    t1 = time.time()
    for i in xrange(n_iters):
        returns.append(agent.train_batch(env, total_steps=2000, batch_size=None)[:2])
        # print agent.session.run(agent.train_graph.learning_rate)
        m_return = returns[-1][0]
        m_ep_len = returns[-1][1]
        print "Iteration %s:" % i
        print "  average return {}\n  average episode length {}".format(m_return, m_ep_len)

        if i % args.evaluate_freq == 0:
            evaluate(env, agent, 5, args.max_steps)
        
    t2 = time.time()
    print '{} sec used, {} sec per iteration.'.format(t2 - t1, (t2 - t1) / n_iters)

    if args.record:
        env.monitor.close()
    
    plt.plot(iter_num, [r[0] for r in returns])
    plt.xlabel('Number of iterations')
    plt.ylabel('Average return')
    plt.show()
    plt.plot(iter_num, [r[1] for r in returns])
    plt.ylabel('Average episode length')
    plt.show()


def evaluate(env, agent, n_eps, max_steps):
    for i_episode in range(n_eps):
        observation = env.reset()
        for t in range(max_steps):
            env.render()
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        else:
            print("Episode reached maximum length  after {} timesteps".format(t+1))


if __name__ == '__main__':
    main()
