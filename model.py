"""Attention-GRU model for microsleep forecasting."""

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Binary focal loss for imbalanced microsleep forecasting."""
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
        p_t = tf.where(tf.equal(y_true_f, 1.0), y_pred_f, 1.0 - y_pred_f)
        alpha_t = tf.where(tf.equal(y_true_f, 1.0), alpha, 1.0 - alpha)
        loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t + epsilon)
        return tf.reduce_mean(loss)
    return loss_fn


class CausalTemporalAttention(layers.Layer):
    """Causal multi-head self-attention before GRU."""

    def __init__(self, attention_dim: int = 64, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

    def build(self, input_shape):
        self.query_dense = layers.Dense(self.attention_dim, use_bias=False)
        self.key_dense = layers.Dense(self.attention_dim, use_bias=False)
        self.value_dense = layers.Dense(self.attention_dim, use_bias=False)
        self.output_dense = layers.Dense(input_shape[-1])
        self.position = self.add_weight(
            name="positional_encoding",
            shape=(input_shape[1], input_shape[-1]),
            initializer="random_normal",
            trainable=True,
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = models.Sequential([
            layers.Dense(self.attention_dim * 2, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(input_shape[-1]),
        ])
        super().build(input_shape)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x_pos = x + self.position[:seq_len, :]

        q = self.query_dense(x_pos)
        k = self.key_dense(x_pos)
        v = self.value_dense(x_pos)

        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        scores = scores + (1.0 - causal_mask) * -1e9
        weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(weights, v)

        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, (batch_size, seq_len, self.attention_dim))
        attended = self.output_dense(attended)

        x = self.norm1(x + attended)
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + ffn_out)


class AttentionPooling(layers.Layer):
    """Post-GRU attention pooling."""
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.w) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)


def build_attention_gru(input_shape, learning_rate: float = 0.001):
    """Build the proposed Attention-GRU model."""
    inputs = layers.Input(shape=input_shape)
    x = CausalTemporalAttention(attention_dim=64, num_heads=4, name="causal_attention")(inputs)
    x = layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = layers.BatchNormalization()(x)

    attn_pool = AttentionPooling(name="post_gru_attention")(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([attn_pool, max_pool])

    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="Attention_GRU_Microsleep_Forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
        ],
    )
    return model
