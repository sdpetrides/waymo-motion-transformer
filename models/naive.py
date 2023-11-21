import time

from functools import reduce

import tensorflow as tf
import keras_nlp

from .layers.fast_attention import SelfAttention as FastSelfAttention
from .layers.fast_attention import Attention as FastCrossAttention


class NaiveModel(tf.keras.Model):
    """A special model designed by Steve."""

    def __init__(
        self,
        num_agents_per_scenario,
        num_state_steps,
        num_future_steps,
        future_step_interval=2,
        rg_interval=10,
        N_rg_encoder=2,
        N_obj_encoder=2,
        N_decoder=4,
        use_performers=True,
        training=True,
    ):
        super(NaiveModel, self).__init__()
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps - 1
        self._num_future_steps = num_future_steps
        self._future_step_interval = future_step_interval
        self._rg_interval = rg_interval
        self.use_performers = use_performers
        self.training = training
        self.N_rg_encoder = N_rg_encoder
        self.N_obj_encoder = N_obj_encoder
        self.N_decoder = N_decoder
        self.init_encoder()
        self.init_decoder()

    def init_encoder(self):
        # Road Graph Encoder
        self.rg_H = 16
        self.rg_out = 1024
        self.rg_dense = tf.keras.layers.Dense(self.rg_H, activation="relu")
        self.rg_qkv_expander = tf.keras.layers.Dense(self.rg_H * 3, activation="relu")
        self.rg_attn_blocks = []
        self.rg_fc_blocks = []
        self.rg_layer_norm_block_1 = []
        self.rg_layer_norm_block_2 = []
        self.rg_fc_out = tf.keras.layers.Dense(self.rg_out, activation="relu")
        for _ in range(self.N_rg_encoder):
            if self.use_performers:
                self.rg_attn_blocks.append(
                    FastSelfAttention(
                        hidden_size=self.rg_H,
                        num_heads=4,
                        attention_dropout=0.2,
                    )
                )
            else:
                self.rg_attn_blocks.append(
                    tf.keras.layers.MultiHeadAttention(
                        int(self.rg_H / 4),
                        key_dim=4,
                        value_dim=4,
                        dropout=0.0,
                        use_bias=True,
                        output_shape=None,
                        attention_axes=(1, 2),
                    )
                )
            self.rg_layer_norm_block_1.append(
                tf.keras.layers.LayerNormalization(axis=-1)
            )
            self.rg_layer_norm_block_2.append(
                tf.keras.layers.LayerNormalization(axis=-1)
            )
            self.rg_fc_blocks.append(
                tf.keras.layers.Dense(self.rg_H, activation="relu")
            )

        # Agent Encoder
        self.obj_H = 1024
        self.obj_dense = tf.keras.layers.Dense(self.obj_H, activation="relu")
        self.obj_positional_encoder = keras_nlp.layers.PositionEmbedding(
            self._num_state_steps, initializer="glorot_uniform"
        )
        self.obj_qkv_expander = tf.keras.layers.Dense(self.obj_H * 3, activation="relu")
        self.obj_attn_blocks = []
        self.obj_fc_blocks = []
        self.obj_layer_norm_block_1 = []
        self.obj_layer_norm_block_2 = []
        for _ in range(self.N_obj_encoder):
            if self.use_performers:
                self.obj_attn_blocks.append(
                    FastCrossAttention(
                        hidden_size=self.obj_H,
                        num_heads=16,
                        attention_dropout=0.2,
                    )
                )
            else:
                self.obj_attn_blocks.append(
                    tf.keras.layers.MultiHeadAttention(
                        int(self.obj_H / 64),
                        key_dim=64,
                        value_dim=64,
                        dropout=0.0,
                        use_bias=True,
                        output_shape=None,
                        attention_axes=(1, 2),
                    )
                )
            self.obj_layer_norm_block_1.append(
                tf.keras.layers.LayerNormalization(axis=-1)
            )
            self.obj_layer_norm_block_2.append(
                tf.keras.layers.LayerNormalization(axis=-1)
            )
            self.obj_fc_blocks.append(
                tf.keras.layers.Dense(self.obj_H, activation="relu")
            )

    def init_decoder(self):
        self.N_decoder = 3  # number of decoder blocks
        self.T = int(self._num_future_steps / self._future_step_interval) + 1
        self.mtn_positional_encoder = keras_nlp.layers.PositionEmbedding(
            self._num_future_steps, initializer="glorot_uniform"
        )
        self.mtn_dense1 = tf.keras.layers.Dense(768, activation="relu")
        self.mtn_dense2 = tf.keras.layers.Dense(1024, activation="relu")
        self.mtn_pool = tf.keras.layers.MaxPooling1D(
            pool_size=4, data_format="channels_first"
        )
        self.mtn_qkv_expander = tf.keras.layers.Dense(768 * 3, activation="relu")
        self.mtn_flatten = tf.keras.layers.Reshape((self.T * 192,))
        self.mtn_masked_attn_blocks = []
        self.mtn_cross_attn_blocks = []
        self.mtn_fc_blocks = []
        self.mtn_layer_norm_block_1 = []
        self.mtn_layer_norm_block_2 = []
        self.mtn_layer_norm_block_3 = []
        for _ in range(self.N_decoder):
            self.mtn_masked_attn_blocks.append(
                tf.keras.layers.MultiHeadAttention(
                    12,
                    key_dim=64,
                    value_dim=64,
                    dropout=0.0,
                    use_bias=True,
                    output_shape=None,
                    attention_axes=None,  # was getting expand_dim error so switched back to None
                )
            )
            if self.use_performers:
                self.mtn_cross_attn_blocks.append(
                    FastCrossAttention(
                        hidden_size=768,
                        num_heads=12,
                        attention_dropout=0.2,
                    )
                )
            else:
                self.mtn_cross_attn_blocks.append(
                    tf.keras.layers.MultiHeadAttention(
                        12,
                        key_dim=64,
                        value_dim=64,
                        dropout=0.0,
                        use_bias=True,
                        output_shape=None,
                        attention_axes=(1, 2),
                    )
                )
            self.mtn_layer_norm_block_1.append(
                tf.keras.layers.LayerNormalization(axis=-1)
            )
            self.mtn_layer_norm_block_2.append(
                tf.keras.layers.LayerNormalization(axis=-1)
            )
            self.mtn_layer_norm_block_3.append(
                tf.keras.layers.LayerNormalization(axis=-1)
            )
            self.mtn_fc_blocks.append(tf.keras.layers.Dense(768, activation="relu"))

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
        scene_output = self.scene_encoder(obj_inputs, road_graph)
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
            next_pred = self.motion_decoder(
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

    def scene_encoder(self, obj_inputs, road_inputs):
        B, T, V = obj_inputs.shape

        #
        # Road Graph Encoder
        #

        # Input Embedding
        x = self.rg_dense(road_inputs)  # B, rg_P, rg_H
        # (no positional encoding for now)
        # Road Graph Encoder
        for i in range(self.N_rg_encoder):
            # Expand to q, k, v
            qkv = self.rg_qkv_expander(x)
            q, k, v = (
                qkv[..., : self.rg_H],
                qkv[..., self.rg_H : -self.rg_H],
                qkv[..., -self.rg_H :],
            )
            # Multi-Head Self-Attention
            if self.use_performers:
                attn_output = self.rg_attn_blocks[i](
                    q, bias=None, training=self.training
                )
            else:
                attn_output = self.rg_attn_blocks[i](q, v, k, use_causal_mask=False)
            # Add & Norm (first time)
            norm_output = self.rg_layer_norm_block_1[i](attn_output + x)
            # Feed Forward
            x = self.rg_fc_blocks[i](norm_output)
            # Add & Norm (second time)
            x = self.rg_layer_norm_block_2[i](norm_output + x)

        rg_out = self.rg_fc_out(x)  # B, rg_P, H

        #
        # Agent Encoder
        #

        # Input Embedding
        x = self.obj_dense(obj_inputs)  # B, T, H
        # Positional Encoding
        x += self.obj_positional_encoder(x)
        # Encoder blocks
        for i in range(self.N_obj_encoder):
            # Expand to q, k, v
            # qkv = self.qkv_expander(x)  # B, T, H * 3
            # q, k, v = (
            #     qkv[..., : self.H],
            #     qkv[..., self.H : -self.H],
            #     qkv[..., -self.H :],
            # )
            q = x
            k, v = rg_out, rg_out
            # Multi-Head Cross-Attention
            if self.use_performers:
                attn_output = self.obj_attn_blocks[i](
                    q, k, bias=None, training=self.training
                )
            else:
                attn_output = self.obj_attn_blocks[i](q, v, k)
            # Add & Norm (first time)
            norm_output = self.obj_layer_norm_block_1[i](attn_output + x)
            # Feed Forward
            x = self.obj_fc_blocks[i](norm_output)
            # Add & Norm (second time)
            x = self.obj_layer_norm_block_2[i](norm_output + x)  # B, T, H

        return x

    def motion_decoder(self, inputs, scene_output, causal_mask):
        # Input embedding
        x = self.mtn_dense1(inputs)  # B, T, 1024 -> B, T, 768
        # Positional Encoding
        x += self.mtn_positional_encoder(x)
        for i in range(self.N_decoder):
            # Expand to q, k, v
            qkv = self.mtn_qkv_expander(x)  # B, T, 768 * 3
            q, k, v = qkv[..., :768], qkv[..., 768:-768], qkv[..., -768:]
            # Masked Multi-Head Self-Attention
            attn_output_1 = self.mtn_masked_attn_blocks[i](
                q, v, k, attention_mask=causal_mask, use_causal_mask=True
            )
            # Add & Norm (first time)
            q = self.mtn_layer_norm_block_1[i](attn_output_1 + x)
            # Multi-Head Cross-Attention
            k, v = scene_output, scene_output
            if self.use_performers:
                attn_output_2 = self.mtn_cross_attn_blocks[i](
                    q, k, bias=None, training=self.training
                )
            else:
                attn_output_2 = self.mtn_cross_attn_blocks[i](q, v, k)
            # Add & Norm (second time)
            norm_output = self.mtn_layer_norm_block_2[i](attn_output_2 + x)
            # Feed Forward
            x = self.mtn_fc_blocks[i](norm_output)
            # Add & Norm (third time)
            x = self.mtn_layer_norm_block_3[i](norm_output + x)

        x = self.mtn_pool(x)  # B, T, 768 -> B, T, 192
        x = self.mtn_flatten(x)  # B, T, 192 -> B, T * 192
        x = self.mtn_dense2(x)  # B, T * 192 -> B, 1024
        return x

    @staticmethod
    def pre_encode(states):
        # Transpose states so that dim order is temporal, object, value
        states = tf.transpose(states, perm=[0, 2, 1, 3])  # (B, T, Obj, V)
        # Flatten each state
        states = tf.reshape(
            states, (*states.shape[:2], reduce(lambda a, b: a * b, states.shape[2:]))
        )  # (B, T, Obj * (V + 1))
        # states = tf.reshape(states, (states.shape[0], reduce(lambda a, b: a * b, states.shape[1:])))  # (B, T * Obj * (V + 1))
        return states

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
