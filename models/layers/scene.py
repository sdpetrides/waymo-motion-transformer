from functools import reduce

import tensorflow as tf
import keras_nlp

from .fast_attention import SelfAttention as FastSelfAttention
from .fast_attention import Attention as FastCrossAttention


class SceneEncoder(tf.keras.layers.Layer):
    """Scene encoder takes the observed scene
    and outputs some latent embedding of the scene.
    """

    def __init__(
        self,
        num_agents_per_scenario,
        num_state_steps,
        use_performers=True,
        training=True,
    ):
        super(SceneEncoder, self).__init__()
        self.N = 2  # number of encoder blocks
        self.H = 1024
        self.use_performers = use_performers
        self.training = training
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps
        self.positional_encoder = keras_nlp.layers.PositionEmbedding(
            self._num_state_steps, initializer="glorot_uniform"
        )
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
        for _ in range(self.N):
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
        self.dense = tf.keras.layers.Dense(self.H, activation="relu")
        self.qkv_expander = tf.keras.layers.Dense(self.H * 3, activation="relu")
        self.attn_blocks = []
        self.fc_blocks = []
        self.layer_norm_block_1 = []
        self.layer_norm_block_2 = []
        for _ in range(self.N):
            if self.use_performers:
                self.attn_blocks.append(
                    FastCrossAttention(
                        hidden_size=self.H,
                        num_heads=16,
                        attention_dropout=0.2,
                    )
                )
            else:
                self.attn_blocks.append(
                    tf.keras.layers.MultiHeadAttention(
                        int(self.H / 64),
                        key_dim=64,
                        value_dim=64,
                        dropout=0.0,
                        use_bias=True,
                        output_shape=None,
                        attention_axes=(1, 2),
                    )
                )
            self.layer_norm_block_1.append(tf.keras.layers.LayerNormalization(axis=-1))
            self.layer_norm_block_2.append(tf.keras.layers.LayerNormalization(axis=-1))
            self.fc_blocks.append(tf.keras.layers.Dense(self.H, activation="relu"))

    def call(self, obj_inputs, road_inputs):
        B, T, V = obj_inputs.shape

        #
        # Road Graph Encoder
        #

        # Input Embedding
        x = self.rg_dense(road_inputs)  # B, rg_P, rg_H
        # (no positional encoding for now)
        # Road Graph Encoder
        for i in range(self.N):
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
            x = self.rg_layer_norm_block_1[i](norm_output + x)

        rg_out = self.rg_fc_out(x)  # B, rg_P, H

        #
        # Agent Encoder
        #

        # Input Embedding
        x = self.dense(obj_inputs)  # B, T, H
        # Positional Encoding
        x += self.positional_encoder(x)
        # Encoder blocks
        for i in range(self.N):
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
                attn_output = self.attn_blocks[i](
                    q, k, bias=None, training=self.training
                )
            else:
                attn_output = self.attn_blocks[i](q, v, k)
            # Add & Norm (first time)
            norm_output = self.layer_norm_block_1[i](attn_output + x)
            # Feed Forward
            x = self.fc_blocks[i](norm_output)
            # Add & Norm (second time)
            x = self.layer_norm_block_1[i](norm_output + x)  # B, T, H
        return x

    def pre_encoder(self, states):
        # Transpose states so that dim order is temporal, object, value
        states = tf.transpose(states, perm=[0, 2, 1, 3])  # (B, T, Obj, V)
        # Flatten each state
        states = tf.reshape(
            states, (*states.shape[:2], reduce(lambda a, b: a * b, states.shape[2:]))
        )  # (B, T, Obj * (V + 1))
        # states = tf.reshape(states, (states.shape[0], reduce(lambda a, b: a * b, states.shape[1:])))  # (B, T * Obj * (V + 1))
        return states
