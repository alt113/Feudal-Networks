
import gym 
import unittest
import tensorflow as tf

from scripts.training.feudal_networks.algos.feudal_policy_optimizer import FeudalPolicyOptimizer

class TestFeudalPolicyOptimizer(unittest.TestCase):

    def test_init(self):
        env = gym.make('OneRoundDeterministicRewardBoxObs-v0')
        with tf.Session() as session:
            feudal_opt = FeudalPolicyOptimizer(env, 0, 'feudal', False)

if __name__ == '__main__':
    unittest.main()
