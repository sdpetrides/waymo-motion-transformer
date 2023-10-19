from functools import reduce

import tensorflow as tf


class SteveModel(tf.keras.Model):
    """A special model designed by Steve."""

    def __init__(self, num_agents_per_scenario, num_state_steps, num_future_steps):
        super(SteveModel, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps
        self._num_future_steps = num_future_steps

    def call(self, states, is_valid):
        """Forward pass of model.

        B is batch dim, Obj is object dim, T is temporal dim.
        """
        B, Obj, T, V = states.shape
        input = self.pre_encoder(states, is_valid)

        # Positional Encoding

        # Encoder Layers x N
        # Multi-head self-attention
        # Feed-forward

        # Decoder Layers X N
        # Masked Multi-head attention
        # Multi-head cross-attention
        # Feed-forward

        # Dense network

        # Single timeslice
        pred = tf.ones((B, Obj, V))
        return pred

    def pre_encoder(self, states, is_valid):
        # Cast bool to float32
        is_valid = tf.cast(is_valid, tf.float32)  # (B, Obj, T)
        # Unflatten
        is_valid = tf.expand_dims(is_valid, axis=-1)  # (B, Obj, T, 1)
        # Add to valid flag to states
        states = tf.concat([states, is_valid], axis=-1)  # (B, Obj, T, V + 1)
        # Transpose states so that dim order is temporal, object, value
        states = tf.transpose(states, perm=[0, 2, 1, 3])  # (B, T, Obj, V + 1)
        # Flatten each state
        states = tf.reshape(
            states, (*states.shape[:2], reduce(lambda a, b: a * b, states.shape[2:]))
        )  # (B, T, Obj * (V + 1))
        # states = tf.reshape(states, (states.shape[0], reduce(lambda a, b: a * b, states.shape[1:])))  # (B, T * Obj * (V + 1))
        return states
