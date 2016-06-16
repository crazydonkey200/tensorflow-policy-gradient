import numpy as np
import tensorflow as tf


def create_rnn_ops(inputs, input_size, rnn_model='lstm',
                   hidden_size=128, num_layers=2, use_dropout=False,
                   dropout_rate=0.0, time_major=True):
    "Utility function to create multi-layer RNN."
    if rnn_model == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif rnn_model == 'lstm':
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    elif rnn_model == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell

    params = {'input_size': input_size}
    if rnn_model == 'lstm':
        # add bias to forget gate in lstm.
        params['forget_bias'] = 0.0

    # Create multilayer cell.
    cell = cell_fn(hidden_size,
                   **params)
    cells = [cell]
    params['input_size'] = hidden_size
    # more explicit way to create cells for MultiRNNCell than
    # [higher_layer_cell] * (self.num_layers - 1)
    for i in range(num_layers-1):
        higher_layer_cell = cell_fn(hidden_size,
                                    **params)
        cells.append(higher_layer_cell)

    if use_dropout and (dropout_rate > 0.0):
        # dropout_rate = tf.placeholder(tf.float32, [], 'dropout_rate')
        cells = [tf.nn.rnn_cell.DropoutWrapper(
            cell, output_keep_prob=1.0-dropout_rate)
                 for cell in cells]
        
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # batch_size = tf.placeholder(tf.int32,
    #                             name='batch_size')

    batch_size = tf.shape(inputs)[1]

    with tf.name_scope('initial_state'):
        # zero_state is used to compute the intial state for cell.
        zero_state = multi_cell.zero_state(batch_size, tf.float32)
        # Placeholder to feed in initial state.
        initial_state = tf.placeholder(tf.float32,
                                       [None, multi_cell.state_size],
                                       'initial_state')

    seq_lens = tf.placeholder(tf.int64, None, 'sequence_lengths')

    outputs, final_state = tf.nn.dynamic_rnn(multi_cell, inputs, seq_lens,
                                             initial_state=initial_state,
                                             time_major=time_major)

    return ((zero_state, initial_state, seq_lens),
            (outputs, final_state))


def create_fnn_ops(inputs, input_dim,
                   hidden_sizes, activation_fns,
                   l2_scale=0.0):
    "Utility function to create multi-layer FNN with l2 regularization."
    x_dim = input_dim
    x = inputs
    for i, h in enumerate(hidden_sizes):
        if activation_fns[i] == tf.nn.relu:
            init_b = 0.1
        else:
            init_b = 0.0

        a = tf.contrib.layers.fully_connected(
            x, h, activation_fn=activation_fns[i],
            weight_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            bias_init=tf.constant_initializer(value=init_b),
            weight_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
        x = a
    outputs = a
    return outputs


def create_softmax_ops(inputs, input_dim, n_classes, use_softmax_bias=True):
    "Uitlity function to create softmax operations."
    with tf.name_scope('Softmax'):
        softmax_w = tf.get_variable("weights", #[output_dim, self.n_actions],
                                    initializer=tf.zeros_initializer(
                                        [input_dim, n_classes]))
        
        if use_softmax_bias:
            softmax_b = tf.get_variable(
                "bias", #[1, self.n_actions],
                initializer=tf.zeros_initializer([n_classes]))
            logits = tf.matmul(inputs, softmax_w) + softmax_b
        else:
            logits = tf.matmul(inputs, softmax_w)

        probs = tf.nn.softmax(logits)

    return logits, probs


def weighted_neg_ll(logits, labels, example_weights):
    with tf.name_scope('weighted_neg_ll'):
        # Compute mean cross entropy loss for each output.
        neg_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels)
        mean_weighted_neg_ll = tf.reduce_mean(
            neg_log_likelihood * example_weights)
    return mean_weighted_neg_ll


def rnn_weighted_neg_ll(logits, labels, example_weights, seq_lens):
    with tf.name_scope('rnn_weighted_neg_ll'):
        # Compute mean cross entropy loss for each output.
        neg_log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels)

        mean_weighted_neg_ll = (tf.reduce_sum(neg_log_likelihood * example_weights) /
                                tf.to_float(tf.reduce_sum(seq_lens)))

    return mean_weighted_neg_ll
