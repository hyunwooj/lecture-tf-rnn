from __future__ import print_function
try:
    from builtins import map, range
except:
    print('pip install future')

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from model import CharRNN


# Config
# python main.py --data=linux --hid_size=128
# python main.py --help
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shakespeare',
                    choices=['linux', 'shakespeare', 'warpeace'])
parser.add_argument('--hid_size', type=int, default=512)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--logdir', type=str, default='logs')
parser.add_argument('--log_step', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--save_model_secs', type=int, default=60)

config = parser.parse_args()

# Data
fname = os.path.join('data', config.data+'_input.txt')
if not os.path.exists(fname):
    print("""Can't find {fname}. Run download_data.sh"""
          """ to download data.""".format(fname=fname))
    sys.exit(1)

with open(fname) as lines:
    corpus = ''.join(list(lines))
    corpus = corpus[:-((len(corpus)-1) % config.seq_len)]  # [seq_len*N+1]

num_seq = len(corpus)//config.seq_len
chars = set(corpus)
corpus_size, num_chars = len(corpus), len(chars)

# char -> idx
char2idx = {c: i for i, c in enumerate(chars)}
# idx -> char
idx2char = {i: c for i, c in enumerate(chars)}

corpus = list(map(char2idx.get, corpus))

with tf.variable_scope('charrnn'):
    # For training
    model = CharRNN(seq_len=config.seq_len,
                    hid_size=config.hid_size,
                    num_layers=config.num_layers,
                    num_chars=num_chars,
                    teacher_forcing=True,
                    reuse=False)

    # Optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(config.lr)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    clipped = [(tf.clip_by_value(grad, -5, 5), var)
               for grad, var in grads_and_vars]
    train_op = optimizer.apply_gradients(clipped, global_step)

    tf.get_variable_scope().reuse_variables()

    # For test
    model_test = CharRNN(seq_len=2*config.seq_len,
                         hid_size=config.hid_size,
                         num_layers=config.num_layers,
                         num_chars=num_chars,
                         teacher_forcing=False,
                         reuse=True)

sv = tf.train.Supervisor(logdir=config.logdir,
                         save_model_secs=config.save_model_secs)

with sv.managed_session() as sess:
    for _ in range(config.num_epochs):
        state = np.zeros([config.num_layers, 2, 1, config.hid_size])
        for step in range(num_seq):
            fetches = [model.output, model.loss, model.last_state,
                       global_step, train_op]
            start, end = step*config.seq_len, (step+1)*config.seq_len
            feed_dict = {
                model.input_: corpus[start:end],
                model.target: corpus[start+1:end+1],
                model.init_state: state,
            }
            output, loss, state, g_step, _ = sess.run(fetches, feed_dict)
            if g_step % config.log_step == 0:
                print('Step:', g_step, 'Loss:', loss)
            if g_step % (config.log_step*10) == 0:
                fetches = [model_test.output, model_test.loss]
                feed_dict = {
                    model_test.input_: corpus[start:start+2*config.seq_len],
                    model_test.target: corpus[start+1:start+2*config.seq_len+1],
                    model_test.init_state: state,
                }
                result = sess.run(fetches, feed_dict)
                output, loss = result
                print('----------------------------------')
                print('Output:')
                print(''.join(map(idx2char.get, output)))
                print('----------------------------------')
