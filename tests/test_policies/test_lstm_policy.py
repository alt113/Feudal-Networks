import unittest

from scripts.training.feudal_networks.policies.lstm_policy import LSTMPolicy
import tensorflow as tf

class TestLSTMPolicy(unittest.TestCase):
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
        global_step = tf.get_variable("global_step", [], tf.int32,\
                                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                                        trainable=False)
        lstm_pi = LSTMPolicy((80,80,3), 4,global_step)

if __name__ == '__main__':
    unittest.main()
