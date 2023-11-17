import time

from functools import reduce

import tensorflow as tf

from .layers.scene import SceneEncoder
from .layers.motion import MotionDecoder


class NaiveModel(tf.keras.Model):
    """A special model designed by Steve."""

    def __init__(
        self,
        num_agents_per_scenario,
        num_state_steps,
        num_future_steps,
        use_performers=True,
        training=True,
    ):
        super(NaiveModel, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps
        self._num_future_steps = num_future_steps
        self.use_performers = use_performers
        self.training = training

        self.scene_encoder = SceneEncoder(
            self._num_agents_per_scenario,
            self._num_state_steps - 1,  # past only
            self.use_performers,
            self.training,
        )
        self.motion_decoder = MotionDecoder(
            self._num_agents_per_scenario,
            self._num_state_steps - 1,  # past only
            self._num_future_steps,
            self.use_performers,
            self.training,
        )
        self.latent_loss_layer = tf.keras.layers.Dense(
            32,
            activation=None,
            trainable=False,
        )

    def call(self, obj_inputs, road_graph):
        """Forward pass of model.

        B is batch dim, T is temporal dim, H is the hidden dim.
        """
        B, T, H = obj_inputs.shape

        obj_inputs = obj_inputs[:, :10, :]  # only use past, not present for now
        road_graph = road_graph[:, ::50, :]  # sample every 50 points

        # Scene Encoder
        start_time = time.time()
        scene_ouput = self.scene_encoder.call(obj_inputs, road_graph)
        print(f"Encoder: {time.time() - start_time}")

        # Decoder Layers (auto-regressive)
        present = obj_inputs[:, -1, :]
        future_states = tf.ones(
            (present.shape[0], self._num_future_steps, present.shape[-1])
        )
        future_states = tf.concat(
            (future_states, tf.expand_dims(present, axis=1)), axis=1
        )
        start_time = time.time()
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
        print(f"Decoder: {time.time() - start_time}")

        return future_states

    def encode(self, states):
        return self.scene_encoder.pre_encoder(states)

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
