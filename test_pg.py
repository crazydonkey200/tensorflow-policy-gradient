import json
import unittest
import numpy as np
import gym

import policy_gradient as pg

class TestPG(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('FrozenLake-v0')
        self.agent = pg.NNAgent(self.env.action_space,
                                self.env.observation_space,
                                max_steps=100, learning_rate=100.0,
                                discount=0.98)

    def test_get_batch(self):
        paths, mean_return, mean_ep_len = self.agent.get_batch(
            self.env, total_steps=2000)
        self.assertTrue(np.sum([path['ep_len'] for path in paths]) <= 2000)


if __name__ == '__main__':
    unittest.main()
