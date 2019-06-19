
import numpy as np
import tensorflow as tf

def flatten(x):
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
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def categorical_sample(logits, d):
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
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(
        logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)
