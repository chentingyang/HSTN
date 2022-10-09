import tensorflow as tf
import numpy as np
import utils
import utils.metrics as Metrics
import utils.metrics_seq as Metrics_seq
import random
from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.python.keras.callbacks import LearningRateScheduler

# arguments
N = 172
# odmax=241

Batch_Size = 64
Weather_Dim = 13

np.random.seed(100)
random.seed(100)


###################################################################

class GraphConvolution(tf.keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        # assert len(features_shape) == 2
        input_dim = features_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(GraphConvolution, self).build(input_shapes)

    def call(self, inputs, mask=None):
        features = inputs[0]
        links = inputs[1]

        result = K.batch_dot(links, features, axes=[2, 1])
        output = K.dot(result, self.kernel)

        if self.bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shapes):
        input_shape = input_shapes[0]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


def Sequence_GCN(data, neighbor, out_dim, act='relu', **kwargs):
    '''
    input: 
        data: the sequence of input OD matrix batch, (batch_size, seq_len, N, N)
        out_dim: GCN out dim
        neighbor : dynamic or geographic semantic OD neighbor, (bs, seq_len, N, N)

    output:
        the sequnce of GCN_out_batch: (batch_size, seq_len, N, out_dim)                         

    '''
    GCN = GraphConvolution(out_dim, activation=act, **kwargs)

    embed = []

    for n in range(data.shape[1]):
        graph = data[:, n, :, :]

        adj = neighbor[:, n, :, :]

        embed.append(GCN([graph, adj]))

    output = tf.stack(embed, axis=1)
    return output


def scaled_dot_product_attention(q, k, v):
    """
        @q, k, v  (batch_size, num_heads, seq_len_q(k,v), depth) 
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        '''
        k = self.wk(k)
        v = self.wv(v)
        '''
        k = self.wq(k)
        v = self.wq(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        concat_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(concat_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, diff):
    return tf.keras.Sequential([tf.keras.layers.Conv1D(filters=diff, kernel_size=3, padding="SAME", activation='relu'),
                                tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding="SAME",
                                                       activation='relu')])


class AttnLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, diff, num_heads, rate=0.3):
        super(AttnLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, diff)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        attn_output, _ = self.mha.call(x, x, x)
        attn_dropout = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs=x + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layernorm2(inputs=out1 + ffn_output)

        return out2


class BahdanauAttention(tf.keras.Model):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(tf.keras.Model):

    def __init__(self, N, heads, embedding_dim, rate, timestep):

        super(Encoder, self).__init__()
        self.N = N
        self.dim = embedding_dim
        self.timestep = timestep
        self.rate = rate
        self.heads = heads
        self.attn_layers = [AttnLayer(self.dim, self.N, self.heads, self.rate) for _ in range(2)]
        self.gru = tf.keras.layers.GRU(units=self.dim, recurrent_initializer='glorot_uniform',
                                       return_sequences=True, return_state=True)

    def call(self, oddata, weather, sem_neighbor, geo_neighbor):

        data = oddata

        last_sequence = data[:, -1, :, :]
        last_sequence = tf.reshape(last_sequence, (-1, self.N))

        # inherent relationship unit
        attn_data = tf.keras.layers.Dense(self.dim, activation='relu')(data)
        attn_out = []
        for i in range(self.timestep):
            x = attn_data[:, i, :, :]

            for i in range(2):
                x = self.attn_layers[i](x, training=True)
            attn_out.append(x)

        attn_out = tf.stack(attn_out, axis=1)

        # sequence_GCN
        # adjacency relationship unit
        x1_nebh = Sequence_GCN(data, geo_neighbor, self.dim)
        x2_nebh = Sequence_GCN(x1_nebh, geo_neighbor, self.dim)
        nebh = tf.keras.layers.Dropout(self.rate)(x2_nebh)

        # flow relationship unit
        x1_seman = Sequence_GCN(data, sem_neighbor, self.dim)
        x2_seman = Sequence_GCN(x1_seman, sem_neighbor, self.dim)
        seman = tf.keras.layers.Dropout(self.rate)(x2_seman)

        # concat&fc
        sequence_out = tf.concat([nebh, seman], axis=-1)
        sequence_out = tf.keras.layers.Dense(self.dim)(sequence_out)
        sequence_out = tf.keras.layers.Dropout(self.rate)(sequence_out)

        # weather embedding
        weath = tf.keras.layers.Dense(10, activation='relu')(weather)
        weath = tf.keras.layers.Dense(self.dim * self.N, activation='relu')(weath)
        weath = tf.keras.layers.Dropout(self.rate)(weath)
        weather_data = tf.reshape(weath, (-1, self.timestep, self.N, self.dim))

        # add(GCN_out, weather, attn_out)
        # HSTN performs better on metro data while ignoring the influence of attention
        # embedding = tf.keras.layers.add([sequence_out, weather_data])
        embedding = tf.keras.layers.add([sequence_out, weather_data, attn_out])
        embedding = tf.reshape(tf.transpose(embedding, [0, 2, 1, 3]), (-1, self.timestep, self.dim))

        output, state = self.gru(embedding)
        return output, state, last_sequence

    def init_hidden_state(self):
        return tf.zeros(shape=(Batch_Size, self.dims))

# dynamic learning unit
class Decoder(tf.keras.Model):

    def __init__(self, out_dim, N, rate=0.3):
        super(Decoder, self).__init__()
        self.dim = out_dim
        self.rate = rate
        self.N = N

        self.gru = tf.keras.layers.GRU(self.dim, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # attention
        self.attention = BahdanauAttention(self.dim)

        def call(self, x, periodic_input, dec_state, enc_output):

            context_vector, attention_weights = self.attention.call(dec_state, enc_output)

            # concat&fc
            dense_inp = tf.concat([x, periodic_input], axis=1)
            dense_out = tf.keras.layers.Dense(self.N, activation='relu')(dense_inp)

            gru_inp = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(dense_out, 1)], axis=-1)

            gru_out, state = self.gru(gru_inp)

            gru_out = tf.reshape(gru_out, (-1, gru_out.shape[2]))

            gru_out = tf.keras.layers.Dropout(self.rate)(gru_out)
            output = tf.keras.layers.Dense(self.N, activation='tanh')(gru_out)
            output = tf.reshape(output, (-1, self.N, self.N))

            return output, state, attention_weights


class AttnSeq2Seq(tf.keras.layers.Layer):

    def __init__(self, N, heads, embedding_dim, rate, timestep, out_seq_len, is_seq, loss='MSE'):

        super(AttnSeq2Seq, self).__init__()
        self.N = N
        self.heads = heads
        self.dim = embedding_dim
        self.timestep = timestep
        self.rate = rate
        self.loss = loss
        self.out_seq_len = out_seq_len
        self.is_seq = is_seq
        self.encoder = Encoder(N, self.heads, self.dim, self.rate, self.timestep)
        self.decoder = Decoder(self.dim, N, rate)
        self.attn_layers = [AttnLayer(self.N, self.N, self.heads, self.rate) for _ in
                            range(2)]  

    def call(self):
        data = tf.keras.Input(shape=(self.timestep, self.N, self.N))
        weather = tf.keras.Input(shape=(self.timestep, Weather_Dim))
        sem_neighbor = tf.keras.Input(shape=(self.timestep, self.N, self.N))
        geo_neighbor = tf.keras.Input(shape=(self.timestep, self.N, self.N))

        # static learning unit
        periodic_data = data
        periodic_data = tf.reshape(tf.transpose(periodic_data, [0, 2, 1, 3]), (-1, self.timestep, self.N))

        for i in range(2):
            periodic_data = self.attn_layers[i](periodic_data, training=True)

        periodic_output = tf.reduce_sum(periodic_data, axis=1)

        oddata = data

        enc_output, enc_state, last_seq = self.encoder.call(data, weather, sem_neighbor, geo_neighbor)

        dec_state = enc_state

        dec_input = last_seq

        outlist = []

        # dynamic
        for t in range(self.out_seq_len):
            predictions, dec_state, _ = self.decoder.call(dec_input, periodic_output, dec_state, enc_output)

            outlist.append(predictions)

            dec_input = tf.reshape(predictions, (-1, self.N))

        output = tf.stack(outlist, axis=1)
        if output.shape[1] == 1:  # single-step
            output = output[:, 0, :, :]

        model = tf.keras.Model(inputs=[oddata, weather, sem_neighbor, geo_neighbor], outputs=output)

        if self.is_seq == False:  # single-step
            model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(0.001),
                          metrics=[Metrics.rmse, Metrics.mae,
                                   Metrics.o_rmse, Metrics.o_mae,
                                   Metrics.d_rmse, Metrics.d_mae])
        else:  # multi-step
            model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(0.001),
                          metrics=[Metrics_seq.rmse, Metrics_seq.mae,
                                   Metrics_seq.o_rmse, Metrics_seq.o_mae,
                                   Metrics_seq.d_rmse, Metrics_seq.d_mae])

        return model
