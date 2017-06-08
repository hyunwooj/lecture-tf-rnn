import tensorflow as tf


class CharRNN(object):
    def __init__(self, seq_len, hid_size, num_chars, teacher_forcing,
                 num_layers, reuse=None):
        self.input_ = tf.placeholder(tf.int32, shape=[seq_len], name='input')
        self.target = tf.placeholder(tf.int32, shape=[seq_len], name='target')
        self.init_state = tf.placeholder(tf.float32,
                                         shape=[num_layers, 2, 1, hid_size],
                                         name='init_state')

        cells = [tf.contrib.rnn.BasicLSTMCell(hid_size, reuse=reuse)
                 for _ in range(num_layers)]
        rnn = tf.contrib.rnn.MultiRNNCell(cells)

        # Output layer weights
        W = tf.get_variable(name='W', shape=[hid_size, num_chars],
                            initializer=tf.random_normal_initializer())
        b = tf.get_variable(name='b', shape=[1, num_chars],
                            initializer=tf.random_normal_initializer())

        input_ = tf.one_hot(self.input_, num_chars)  # [seq_len, num_chars]
        target = tf.one_hot(self.target, num_chars)  # [seq_len, num_chars]

        self.output = []  # will be [seq_len]
        state = [tf.contrib.rnn.LSTMStateTuple(self.init_state[i][0],
                                               self.init_state[i][1])
                 for i in range(num_layers)]
        self.loss = 0
        with tf.variable_scope('rnn'):
            for t in range(seq_len):
                if teacher_forcing or t == 0:
                    in_ = input_[t]  # [num_chars]
                else:
                    in_ = tf.one_hot(out, num_chars)  # [num_chars]
                in_ = tf.expand_dims(in_, axis=0)  # [1, num_chars]

                out, state = rnn(in_, state)  # out: [1, num_chars], state: [1, num_layers*hid_size]

                out = tf.matmul(out, W)+b  # [1, num_chars]
                out = tf.reshape(out, [num_chars])  # [num_chars]

                self.loss += tf.nn.softmax_cross_entropy_with_logits(
                    logits=out, labels=target[t])

                out = tf.argmax(out)  # []
                self.output.append(out)  # [t+1]
                tf.get_variable_scope().reuse_variables()  # idempotent
            self.last_state = state
