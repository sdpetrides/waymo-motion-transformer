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
        future_step_interval=2,
        rg_interval=10,
        use_performers=True,
        training=True,
    ):
        super(NaiveModel, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps
        self._num_future_steps = num_future_steps
        self._future_step_interval = future_step_interval
        self._rg_interval = rg_interval
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
            self._future_step_interval,
            self.use_performers,
            self.training,
        )

    def call(self, inputs):
        """Forward pass of model.

        B is batch dim, T is temporal dim, H is the hidden dim.
        """
        obj_inputs, road_graph, future_states_base = inputs
        B, T, H = obj_inputs.shape
        T_decoder = int(self._num_future_steps / self._future_step_interval)

        obj_inputs = obj_inputs[:, :10, :]
        road_graph = road_graph[:, :: self._rg_interval, :]  # sample every 10 points

        # Scene Encoder
        # start_time = time.time()
        scene_output = self.scene_encoder.call(obj_inputs, road_graph)
        # print(f"Encoder: {time.time() - start_time}")

        # Decoder Layers (auto-regressive)
        present = obj_inputs[:, -1, :]
        future_states = tf.concat(
            (future_states_base, tf.expand_dims(present, axis=1)), axis=1
        )
        # start_time = time.time()
        for i in range(1, T_decoder + 1):
            # print(f"Generating step {i}")
            causal_mask = self.create_mask(B, T_decoder + 1, i)
            next_pred = self.motion_decoder.call(
                future_states,
                scene_output,
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
        # print(f"Decoder: {time.time() - start_time}")

        return future_states

    def encode(self, states):
        return self.scene_encoder.pre_encoder(states)

    @staticmethod
    def create_mask(batch_size, matrix_size, k):
        if batch_size == None:
            batch_size = 1
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
