from tensorflow.keras.layers import Layer
import tensorflow as tf

class NetVLAD(Layer):
    def __init__(self, num_clusters, dim, **kwargs):
        super(NetVLAD, self).__init__(**kwargs)
        self.num_clusters = num_clusters
        self.dim = dim

    def build(self, input_shape):
        self.clusters = self.add_weight(
            shape=(self.dim, self.num_clusters),
            initializer='uniform',
            trainable=True,
            name='cluster_centers'
        )
        self.cluster_weights = self.add_weight(
            shape=(self.dim, self.num_clusters),
            initializer='uniform',
            trainable=True,
            name='cluster_weights'
        )
        super(NetVLAD, self).build(input_shape)

    def call(self, inputs):
        s = tf.tensordot(inputs, self.cluster_weights, axes=[2, 0])  # (batch, time, num_clusters)
        a = tf.nn.softmax(s, axis=-1)  # soft-assignment

        a = tf.expand_dims(a, -2)  # (batch, time, 1, num_clusters)
        inputs_expanded = tf.expand_dims(inputs, -1)  # (batch, time, dim, 1)

        vlad = a * (inputs_expanded - tf.expand_dims(self.clusters, 0))  # residual
        vlad = tf.reduce_sum(vlad, axis=1)  # sum over time steps

        vlad = tf.reshape(vlad, [-1, self.num_clusters * self.dim])
        vlad = tf.nn.l2_normalize(vlad, axis=1)
        return vlad

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_clusters * self.dim)
