import tensorflow as tf
"""
This file defines attributes of FLAGS, that are a way of avoiding using argparser, 
defining all hyperparameters in a separate file
"""

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('warmup', 0, 'time without training but only filling the replay memory')
flags.DEFINE_integer('bsize', 64, 'minibatch size')
flags.DEFINE_integer('iter', 1, 'train iters each timestep')
flags.DEFINE_integer('rmsize', 500000, 'memory size')

flags.DEFINE_float('tau', 0.01, 'moving average for target network')
flags.DEFINE_float('alpha', 0.6, 'PER parameter that controls prioritization')
flags.DEFINE_float('discount', 0.99, 'gamma')
flags.DEFINE_float('alpha_beyond', 1e5, 'loss for going beyond rmin,rmax')
flags.DEFINE_float('RMAX', 1, 'reward max')
flags.DEFINE_float('RMIN', 0, 'reward min')
flags.DEFINE_float('a_min', -1, 'action min')
flags.DEFINE_float('a_max', 1, 'action max')
flags.DEFINE_float('l2norm', 0.0001, 'l2 weight decay')
flags.DEFINE_float('rate', 0.001, 'learning rate')
flags.DEFINE_float('outheta', 0.15, 'noise theta')
flags.DEFINE_float('ousigma', 0.1, 'noise sigma')
flags.DEFINE_float('ousigma_end', 0.15, 'noise sigma')  # final stddev on noise on actions (linear decreasing)
flags.DEFINE_float('ousigma_start', 0.5, 'noise sigma')  # start stddev on noise on actions (linear decreasing)
flags.DEFINE_float('lrelu', 0.01, 'leak relu rate')
flags.DEFINE_boolean('icnn_bn', True, 'enable icnn batch normalization')

flags.DEFINE_string('icnn_opt', 'adam',
                    "ICNN's inner optimization routine. Options=[adam,bundle_entropy]")
flags.DEFINE_string('outdir', 'tensorboard', 'where to save files')
flags.DEFINE_integer('thread', 1, 'tensorflow threads')

flags.DEFINE_boolean('summary', True, 'use tensorboard log')

flags.DEFINE_float('initstd', 0.01, 'weight init std')