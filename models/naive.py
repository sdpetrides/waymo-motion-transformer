from functools import reduce

import tensorflow as tf

from .layers.scene import SceneEncoder
from .layers.motion import MotionDecoder


class NaiveModel(tf.keras.Model):
    """A special model designed by Steve."""

    def __init__(self, num_agents_per_scenario, num_state_steps, num_future_steps):
        super(NaiveModel, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps
        self._num_future_steps = num_future_steps

        self.scene_encoder = SceneEncoder(
            self._num_agents_per_scenario,
            self._num_state_steps - 1,  # past only
        )
        self.motion_decoder = MotionDecoder(
            self._num_agents_per_scenario,
            self._num_state_steps - 1,  # past only
            self._num_future_steps,
        )
        self.latent_loss_layer = tf.keras.layers.Dense(
            32,
            activation=None,
            trainable=False,
        )

    def call(self, inputs):
        """Forward pass of model.

        B is batch dim, Obj is object dim, T is temporal dim.
        """
        B, T, H = inputs.shape

        scene_input = inputs[:, :10, :]  # only use past, not present for now

        # Positional Encoding
        scene_input = self.scene_encoder.positional_encoder(scene_input)

        # Scene Encoder
        scene_ouput = self.scene_encoder.call(scene_input)

        # Decoder Layers (auto-regressive)
        present = inputs[:, -1, :]
        future_states = tf.ones(
            (present.shape[0], self._num_future_steps, present.shape[-1])
        )
        future_states = tf.concat(
            (future_states, tf.expand_dims(present, axis=1)), axis=1
        )
        for i in range(1, self._num_future_steps + 1):
            # print(f"Generating step {i}")
            causal_mask = self.create_mask(B, self._num_future_steps, i)
            next_pred = self.motion_decoder.call(
                future_states[:, :-1, :],  # never actually send in last future step
                scene_ouput,
                causal_mask,
            )
            future_states = tf.concat(
                (
                    future_states[:, :i, :],
                    tf.expand_dims(next_pred, axis=1),
                    future_states[:, i + 1 :, :],
                ),
                axis=1,
            )

        return future_states

    def encode(self, states, is_valid):
        return self.scene_encoder.pre_encoder(states, is_valid)

    @staticmethod
    def create_mask(batch_size, matrix_size, k):
        if k < 1 or k > matrix_size:
            raise ValueError(f"k should be within the [1, {matrix_size}] range")
        lower_diag = tf.linalg.band_part(
            tf.ones((batch_size, matrix_size, matrix_size), dtype=tf.bool), -1, 0
        )
        first_k_rows = tf.concat(
            (
                tf.ones((batch_size, k, matrix_size), dtype=tf.bool),
                tf.zeros((batch_size, matrix_size - k, matrix_size), dtype=tf.bool),
            ),
            axis=1,
        )
        return lower_diag & first_k_rows
