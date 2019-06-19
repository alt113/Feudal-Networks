
import gym 
import unittest
import tensorflow as tf

from scripts.training.feudal_networks.algos.feudal_policy_optimizer import FeudalPolicyOptimizer

class TestFeudalPolicyOptimizer(unittest.TestCase):
    """Create a Gym environment by passing environment id.

    Parameters
    ----------
    env_id : str
        environment id to be registered in Gym
    client_id : str
        Client ID
    remotes : str
        BLANK
    kwargs : dict
        BLANK
    """

    def test_init(self):
        """Create a Gym environment by passing environment id.

        Parameters
        ----------
        env_id : str
            environment id to be registered in Gym
        client_id : str
            Client ID
        remotes : str
            BLANK
        kwargs : dict
            BLANK
        """
        env = gym.make('OneRoundDeterministicRewardBoxObs-v0')
        with tf.Session() as session:
            feudal_opt = FeudalPolicyOptimizer(env, 0, 'feudal', False)

if __name__ == '__main__':
    unittest.main()
