from functools import reduce

import tensorflow as tf
import keras_nlp


class SceneEncoder(tf.keras.layers.Layer):
    """Scene encoder takes the observed scene
    and outputs some latent embedding of the scene.
    """

    def __init__(self, num_agents_per_scenario, num_state_steps):
        super(SceneEncoder, self).__init__()
        self.N = 2  # number of encoder blocks
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps
        self.positional_encoder = keras_nlp.layers.PositionEmbedding(
            self._num_state_steps, initializer="glorot_uniform"
        )
        self.dense = tf.keras.layers.Dense(768, activation="relu")
        self.qkv_expander = tf.keras.layers.Dense(768 * 3, activation="relu")
        self.attn_blocks = []
        self.fc_blocks = []
        self.layer_norm_block_1 = []
        self.layer_norm_block_2 = []
        for _ in range(self.N):
            self.attn_blocks.append(
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
            self.layer_norm_block_1.append(tf.keras.layers.LayerNormalization(axis=-1))
            self.layer_norm_block_2.append(tf.keras.layers.LayerNormalization(axis=-1))
            self.fc_blocks.append(tf.keras.layers.Dense(768, activation="relu"))

    def call(self, inputs):
        # Input Embedding
        x = self.dense(inputs)  # B, T, 768
        # Positional Encoding
        x += self.positional_encoder(x)
        # Encoder blocks
        for i in range(self.N):
            # Expand to q, k, v
            qkv = self.qkv_expander(x)  # B, T, 768 * 3
            q, k, v = qkv[..., :768], qkv[..., 768:-768], qkv[..., -768:]
            # Multi-Head Self-Attention
            attn_output = self.attn_blocks[i](q, v, k, use_causal_mask=False)
            # Add & Norm (first time)
            norm_output = self.layer_norm_block_1[i](attn_output + x)
            # Feed Forward
            x = self.fc_blocks[i](norm_output)
            # Add & Norm (second time)
            x = self.layer_norm_block_1[i](norm_output + x)  # B, T, 768
        return x

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
