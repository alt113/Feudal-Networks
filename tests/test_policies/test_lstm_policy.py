import unittest

from scripts.training.feudal_networks.policies.lstm_policy import LSTMPolicy
import tensorflow as tf

class TestLSTMPolicy(unittest.TestCase):

    def test_init(self):
        global_step = tf.get_variable("global_step", [], tf.int32,\
                                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                                        trainable=False)
        lstm_pi = LSTMPolicy((80,80,3), 4,global_step)

if __name__ == '__main__':
    unittest.main()
