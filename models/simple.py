import tensorflow as tf


class SimpleModel(tf.keras.Model):
    """A simple one-layer regressor."""

    def __init__(self, num_agents_per_scenario, num_states_steps, num_future_steps):
        super(SimpleModel, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_states_steps = num_states_steps
        self._num_future_steps = num_future_steps
        self.regressor = tf.keras.layers.Dense(num_future_steps * 2)

    def call(self, states):
        states = tf.reshape(states, (-1, self._num_states_steps * 2))
        pred = self.regressor(states)
        pred = tf.reshape(
            pred, [-1, self._num_agents_per_scenario, self._num_future_steps, 2]
        )
        return pred
