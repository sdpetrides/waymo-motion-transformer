import tensorflow as tf
import keras_nlp


class MotionDecoder(tf.keras.layers.Layer):
    """Motion decoder takes an timeslice and cross-attends
    with observed scene and predicts the next timeslice.
    """

    def __init__(self, num_agents_per_scenario, num_state_steps, num_future_steps):
        super(MotionDecoder, self).__init__()
        self.N = 2  # number of decoder blocks
        self._num_agents_per_scenario = num_agents_per_scenario
        self._num_state_steps = num_state_steps
        self._num_future_steps = num_future_steps
        self.positional_encoder = keras_nlp.layers.PositionEmbedding(
            self._num_future_steps, initializer="glorot_uniform"
        )
        self.dense1 = tf.keras.layers.Dense(768, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1024, activation="relu")
        self.pool = tf.keras.layers.MaxPooling1D(
            pool_size=4, data_format="channels_first"
        )
        self.qkv_expander = tf.keras.layers.Dense(768 * 3, activation="relu")
        self.masked_attn_blocks = []
        self.cross_attn_blocks = []
        self.fc_blocks = []
        self.layer_norm_block_1 = []
        self.layer_norm_block_2 = []
        self.layer_norm_block_3 = []
        for _ in range(self.N):
            self.masked_attn_blocks.append(
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
            self.cross_attn_blocks.append(
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
            self.layer_norm_block_3.append(tf.keras.layers.LayerNormalization(axis=-1))
            self.fc_blocks.append(tf.keras.layers.Dense(768, activation="relu"))

    def call(self, inputs, scene_output, causal_mask):
        """Decoder with masked self-attention and cross-attention.

        Args:
            inputs: Tensor of shape (B, T_full, H)
            scene_output: Tensor of shape (B, H)
            causal_mask: Tensor of shape (B, T_full, H) of ones and zeros

        Return:
            A new motion prediction, Tensor of shape (B, H)
        """
        # Input embedding
        x = self.dense1(inputs)  # B, T, 1024 -> B, T, 768
        # Positional Encoding
        x += self.positional_encoder(x)
        for i in range(self.N):
            # Expand to q, k, v
            qkv = self.qkv_expander(x)  # B, T, 768 * 3
            q, k, v = qkv[..., :768], qkv[..., 768:-768], qkv[..., -768:]
            # Masked Multi-Head Self-Attention
            attn_output_1 = self.masked_attn_blocks[i](
                q, v, k, attention_mask=causal_mask, use_causal_mask=True
            )
            # Add & Norm (first time)
            q = self.layer_norm_block_1[i](attn_output_1 + x)
            # Multi-Head Cross-Attention
            k, v = scene_output, scene_output
            attn_output_2 = self.cross_attn_blocks[i](q, v, k)
            # Add & Norm (second time)
            norm_output = self.layer_norm_block_2[i](attn_output_2 + x)
            # Feed Forward
            x = self.fc_blocks[i](norm_output)
            # Add & Norm (third time)
            x = self.layer_norm_block_3[i](norm_output + x)
        x = self.pool(x)  # B, T, 768 -> B, T, 192
        x = tf.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        x = self.dense2(x)  # B, 80 * 192 -> B, T, 1024
        return x

    def positional_encoder(self, input):
        return input
