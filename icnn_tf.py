"""
This file is a modified version of the code from LocusLab
It implements ICNN using tensorflow
The FLAGS attributes (all hyperparameters) are defined in the agent.py file
"""

import os
import numpy as np
import numpy.random as npr
import tensorflow as tf
import tflearn
from replay_buffer import PrioritizedReplayBuffer
from utils import variable_summaries
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib.mlab import griddata
from sklearn.decomposition import PCA


flags = tf.app.flags
FLAGS = flags.FLAGS


# Input Convex Neural Network
class Agent:
    # @todo: when instantiating two of these, it raises an Exception, because it tries to redefine
    # @todo: the scopes or variables (the names are already taken)
    # @todo: FIX THIS !!!
    """
    We don't use the bundle entropy method to optimize wrt actions,
    but rather plain SGD (or rather Adam)
    """
    def __init__(self, dimO, dimA, beta, layers_dim, finalize_graph=True):
        """
        :param finalize_graph: if you want to restore a model, using .restore(), set this param to False
        """
        self.dimA = dimA
        self.dimO = dimO
        self.beta = beta
        self.layers_dim = layers_dim

        tau = FLAGS.tau
        discount = FLAGS.discount
        l2norm = FLAGS.l2norm
        learning_rate = FLAGS.rate

        self.opt = self.adam

        self.rm = PrioritizedReplayBuffer(FLAGS.rmsize, FLAGS.alpha)
        self.sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=FLAGS.thread,
            log_device_placement=False,
            allow_soft_placement=True))

        self.noise = np.zeros(self.dimA)
        per_weights = tf.placeholder(tf.float32, [None], 'per_weights')

        obs = tf.placeholder(tf.float32, [None, dimO], "obs")
        act = tf.placeholder(tf.float32, [None, dimA], "act")
        rew = tf.placeholder(tf.float32, [None], "rew")
        with tf.variable_scope('q'):
            negQ = self.negQ(obs, act)
        q = -negQ
        act_grad, = tf.gradients(negQ, act)

        obs_target = tf.placeholder(tf.float32, [None, dimO], "obs_target")
        act_target = tf.placeholder(tf.float32, [None, dimA], "act_target")
        term_target = tf.placeholder(tf.bool, [None], "term_target")
        with tf.variable_scope('q_target'):
            negQ_target = self.negQ(obs_target, act_target)
        act_target_grad, = tf.gradients(negQ_target, act_target)
        q_target = -negQ_target

        y = tf.where(term_target, rew, rew + discount * q_target)
        y = tf.maximum(q - 1., y)
        y = tf.minimum(q + 1., y)
        y = tf.stop_gradient(y)
        print('y shape', y.get_shape())
        print('q shape', q.get_shape())
        td_error = q - y
        print('per weights shape', per_weights.get_shape())
        print('multi td error^2 per weights shape', tf.multiply(tf.square(td_error), per_weights).get_shape())
        ms_td_error = tf.reduce_sum(tf.multiply(tf.square(td_error), per_weights), 0)
        print('ms td error shape', ms_td_error.get_shape())

        regLosses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='q/')
        loss_q = ms_td_error + \
                 l2norm * tf.reduce_sum(regLosses) + \
                 FLAGS.alpha_beyond * tf.reduce_sum(
                     tf.where(
                         q > FLAGS.RMAX,
                         tf.square(q - FLAGS.RMAX),
                         tf.zeros((FLAGS.bsize,))) +
                     tf.where(
                         q < FLAGS.RMIN,
                         tf.square(q - FLAGS.RMIN),
                         tf.zeros((FLAGS.bsize,))),
                     0
                 )

        self.theta_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q/')
        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]
        self.makeCvx = [v.assign(tf.abs(v)) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        self.theta_target_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope='q_target/')
        update_target = [theta_target_i.assign_sub(tau * (theta_target_i - theta_i))
                         for theta_i, theta_target_i in zip(self.theta_, self.theta_target_)]

        optim_q = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars_q = optim_q.compute_gradients(loss_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)

        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'board'),
                                               self.sess.graph)
        tf.summary.scalar('Qvalue (batch avg)', tf.reduce_mean(q))
        tf.summary.scalar('Qvalue (batch max)', tf.reduce_max(q))
        tf.summary.scalar('Qvalue (batch min)', tf.reduce_min(q))
        tf.summary.scalar('Q targets (batch avg)', tf.reduce_mean(q_target))
        tf.summary.scalar('Q targets (batch min)', tf.reduce_min(q_target))
        tf.summary.scalar('Q targets (batch max)', tf.reduce_max(q_target))
        tf.summary.scalar('loss', ms_td_error)
        tf.summary.scalar('td error', tf.reduce_mean(tf.abs(td_error)))
        tf.summary.scalar('reward', tf.reduce_mean(rew))
        tf.summary.scalar('chosen actions', tf.reduce_mean(act))
        tf.summary.scalar('maximizing action (batch avg)', tf.reduce_mean(act_target))
        tf.summary.scalar('maximizing action (batch max)', tf.reduce_max(act_target))
        tf.summary.scalar('maximizing action (batch min)', tf.reduce_min(act_target))
        merged = tf.summary.merge_all()

        # tf functions
        with self.sess.as_default():
            self._train = Fun([obs, act, rew, obs_target, act_target, term_target, per_weights],
                              [optimize_q, update_target, loss_q, tf.abs(td_error), q, q_target],
                              merged, summary_writer)
            self._fg = Fun([obs, act], [negQ, act_grad])
            self._fg_target = Fun([obs_target, act_target], [negQ_target, act_target_grad])

        # initialize tf variables
        self.saver = tf.train.Saver(max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.outdir + "/tf")
        if ckpt:
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.makeCvx)
            self.sess.run([theta_target_i.assign(theta_i)
                           for theta_i, theta_target_i in zip(self.theta_, self.theta_target_)])

        if finalize_graph:
            self.sess.graph.finalize()

        self.t = 0  # global training time (number of observations)

    def adam(self, func, obs, plot=False):
        """Optimizer to find the greedy action"""
        # if npr.random() < 1./20:
        #     plot = True
        b1 = 0.9
        b2 = 0.999
        lam = 0.5
        eps = 1e-8
        alpha = 0.01
        nBatch = obs.shape[0]
        act = np.zeros((nBatch, self.dimA))
        m = np.zeros_like(act)
        v = np.zeros_like(act)

        b1t, b2t = 1., 1.
        act_best, a_diff, f_best = [None] * 3
        hist = {'act': [], 'f': [], 'g': []}
        for i in range(1000):
            f, g = func(obs, act)
            if plot:
                hist['act'].append(act.copy())
                hist['f'].append(f)
                hist['g'].append(g)

            if i == 0:
                act_best = act.copy()
                f_best = f.copy()
            else:
                prev_act_best = act_best.copy()
                I = (f < f_best)
                act_best[I] = act[I]
                f_best[I] = f[I]
                a_diff_i = np.mean(np.linalg.norm(act_best - prev_act_best, axis=1))
                a_diff = a_diff_i if a_diff is None \
                    else lam * a_diff + (1. - lam) * a_diff_i
                # print(a_diff_i, a_diff, np.sum(f))
                if a_diff < 1e-3 and i > 5:
                    if plot:
                        self.adam_plot(func, obs, hist)
                    return act_best

            m = b1 * m + (1. - b1) * g
            v = b2 * v + (1. - b2) * (g * g)
            b1t *= b1
            b2t *= b2
            mhat = m / (1. - b1t)
            vhat = v / (1. - b2t)

            act -= alpha * mhat / (np.sqrt(v) + eps)
            act = np.clip(act, FLAGS.a_min + 1e-8, FLAGS.a_max - 1e-8)

        print('  + Warning: Adam did not converge.')
        if plot:
            self.adam_plot(func, obs, hist)
        return act_best

    def adam_plot(self, func, obs, hist):
        hist['act'] = np.array(hist['act']).T
        hist['f'] = np.array(hist['f']).T
        hist['g'] = np.array(hist['g']).T
        if self.dimA == 1:
            xs = np.linspace(-1. + 1e-8, 1. - 1e-8, 100)
            ys = [func(obs[[0], :], [[xi]])[0] for xi in xs]
            fig = plt.figure()
            plt.plot(xs, ys)
            plt.plot(hist['act'][0, 0, :], hist['f'][0, :], label='Adam')
            plt.legend()
            fname = os.path.join(FLAGS.outdir, 'adamPlt.png')
            print("Saving Adam plot to {}".format(fname))
            plt.savefig(fname)
            plt.close(fig)
        elif self.dimA == 2:
            assert (False)
        else:
            xs = npr.uniform(-1., 1., (5000, self.dimA))
            ys = np.array([func(obs[[0], :], [xi])[0] for xi in xs])
            epi = np.hstack((xs, ys))
            pca = PCA(n_components=2).fit(epi)
            W = pca.components_[:, :-1]
            xs_proj = xs.dot(W.T)
            fig = plt.figure()

            X = Y = np.linspace(xs_proj.min(), xs_proj.max(), 100)
            Z = griddata(xs_proj[:, 0], xs_proj[:, 1], ys.ravel(),
                         X, Y, interp='linear')

            plt.contourf(X, Y, Z, 15)
            plt.colorbar()

            adam_x = hist['act'][:, 0, :].T
            adam_x = adam_x.dot(W.T)
            plt.plot(adam_x[:, 0], adam_x[:, 1], label='Adam', color='k')
            plt.legend()

            fname = os.path.join(FLAGS.outdir, 'adamPlt.png')
            print("Saving Adam plot to {}".format(fname))
            plt.savefig(fname)
            plt.close(fig)

    def reset(self, obs):
        self.noise = np.zeros(self.dimA)
        self.observation = obs  # initial observation

    def act(self, test=False):
        """
        Greedily choose action
        There is noise during training
        """
        with self.sess.as_default():
            obs = np.expand_dims(self.observation, axis=0)

            f = self._fg

            tflearn.is_training(False)
            action = self.opt(f, obs)
            tflearn.is_training(not test)

            if not test:
                # sig = (self.t < 40000) * (self.t * (FLAGS.ousigma_end - FLAGS.ousigma_start) / 40000 + FLAGS.ousigma_start) + (self.t >= 40000) * FLAGS.ousigma_end
                # self.noise = sig * npr.randn(self.dimA)
                self.noise -= FLAGS.outheta * self.noise - FLAGS.ousigma * npr.randn(self.dimA)
                action += self.noise
            action = np.clip(action, FLAGS.a_min, FLAGS.a_max)

            self.action = np.atleast_1d(np.squeeze(action, axis=0))
            return self.action

    def observe(self, rew, term, obs2, test=False):
        obs1 = self.observation
        self.observation = obs2

        # train
        if not test:

            self.rm.add(*(obs1, self.action, rew, obs2, term))

            if self.t > FLAGS.warmup:
                for i in range(FLAGS.iter):
                    loss = self.train()

    def train(self):
        self.t += 1
        beta = self.beta(self.t)
        with self.sess.as_default():
            obs, act, rew, ob2, term2, w, idx = self.rm.sample(FLAGS.bsize, beta)
            rew, term2, w = rew.squeeze(), term2.squeeze(), w.squeeze()  # fix dimensions
            # w = np.ones(w.shape)  # no prioritization
            f = self._fg_target
            tflearn.is_training(False)
            act2 = self.opt(f, ob2)
            tflearn.is_training(True)

            _, _, loss, td_error, q, q_target = self._train(obs, act, rew, ob2, act2, term2, w,
                                                            log=FLAGS.summary, global_step=self.t)
            self.sess.run(self.proj)  # keep some weights positive
            # self.rm.update_priorities(idx, np.array(td_error.shape[0] * [1.]))  # no prioritization
            self.rm.update_priorities(idx, td_error + 1e-2)
            return loss, td_error, q, q_target

    def negQ(self, x, y, reuse=False):
        """Architecture of the neural network"""
        print('x shape', x.get_shape())
        print('y shape', y.get_shape())
        szs = self.layers_dim
        assert (len(szs) >= 1)
        fc = tflearn.fully_connected
        bn = tflearn.batch_normalization
        lrelu = tflearn.activations.leaky_relu

        if reuse:
            tf.get_variable_scope().reuse_variables()

        nLayers = len(szs)
        us = []
        zs = []
        z_zs = []
        z_ys = []
        z_us = []

        reg = 'L2'

        prevU = x
        for i in range(nLayers):
            with tf.variable_scope('u' + str(i), reuse=reuse) as s:
                u = fc(prevU, szs[i], reuse=reuse, scope=s, regularizer=reg)
                if i < nLayers - 1:
                    u = tf.nn.relu(u)
                    if FLAGS.icnn_bn:
                        u = bn(u, reuse=reuse, scope=s, name='bn')
            variable_summaries(u, suffix='u{}'.format(i))
            us.append(u)
            prevU = u

        prevU, prevZ = x, y
        for i in range(nLayers + 1):
            sz = szs[i] if i < nLayers else 1
            z_add = []
            if i > 0:
                with tf.variable_scope('z{}_zu_u'.format(i), reuse=reuse) as s:
                    zu_u = fc(prevU, szs[i - 1], reuse=reuse, scope=s,
                              activation='relu', bias=True,
                              regularizer=reg, bias_init=tf.constant_initializer(1.))
                    variable_summaries(zu_u, suffix='zu_u{}'.format(i))
                with tf.variable_scope('z{}_zu_proj'.format(i), reuse=reuse) as s:
                    z_zu = fc(tf.multiply(prevZ, zu_u), sz, reuse=reuse, scope=s,
                              bias=False, regularizer=reg)
                    variable_summaries(z_zu, suffix='z_zu{}'.format(i))
                z_zs.append(z_zu)
                z_add.append(z_zu)

            with tf.variable_scope('z{}_yu_u'.format(i), reuse=reuse) as s:
                yu_u = fc(prevU, self.dimA, reuse=reuse, scope=s, bias=True,
                          regularizer=reg, bias_init=tf.constant_initializer(1.))
                variable_summaries(yu_u, suffix='yu_u{}'.format(i))
            with tf.variable_scope('z{}_yu'.format(i), reuse=reuse) as s:
                z_yu = fc(tf.multiply(y, yu_u), sz, reuse=reuse, scope=s, bias=False,
                          regularizer=reg)
                z_ys.append(z_yu)
                variable_summaries(z_yu, suffix='z_yu{}'.format(i))
            z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(i), reuse=reuse) as s:
                z_u = fc(prevU, sz, reuse=reuse, scope=s,
                         bias=True, regularizer=reg,
                         bias_init=tf.constant_initializer(0.))
                variable_summaries(z_u, suffix='z_u{}'.format(i))
            z_us.append(z_u)
            z_add.append(z_u)

            z = tf.add_n(z_add)
            variable_summaries(z, suffix='z{}_preact'.format(i))
            if i < nLayers:
                # z = tf.nn.relu(z)
                z = lrelu(z, alpha=FLAGS.lrelu)
                variable_summaries(z, suffix='z{}_act'.format(i))

            zs.append(z)
            prevU = us[i] if i < nLayers else None
            prevZ = z

        print('z shape', z.get_shape())
        z = tf.reshape(z, [-1], name='energies')
        return z

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, filename):
        """
        IMPORTANT:
        Filename should be the filepath to the 4 following files:
            - 50314.index
            - 50314.meta
            - 50314.data-00000-of-00001
            - checkpoint
        Note that it shouldn't include any extension. In this case, it would therefore be `tensorboard/models/50314`
        Note that it is `50314` because I used the global training step as a filename to save model

        !!!! BESIDES YOU SHOULD HAVE INSTANTIATED THE AGENT WITH `finalize_graph=False` !!!!
        """
        self.saver = tf.train.import_meta_graph(filename+'.meta')
        self.saver.restore(self.sess, filename)
        self.sess.graph.finalize()

    def __del__(self):
        self.sess.close()


# Tensorflow utils
#
class Fun:
    """ Creates a python function that maps between inputs and outputs in the computational graph. """

    def __init__(self, inputs, outputs, summary_ops=None, summary_writer=None, session=None):
        self._inputs = inputs if type(inputs) == list else [inputs]
        self._outputs = outputs
        # self._summary_op = tf.merge_summary(summary_ops) if type(summary_ops) == list else summary_ops
        self._summary_op = tf.summary.merge(summary_ops) if type(summary_ops) == list else summary_ops
        self._session = session or tf.get_default_session()
        self._writer = summary_writer

    def __call__(self, *args, **kwargs):
        """
        Arguments:
          **kwargs: input values
          log: if True write summary_ops to summary_writer
          global_step: global_step for summary_writer
        """
        log = kwargs.get('log', False)

        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg

        out = self._outputs + [self._summary_op] if log else self._outputs
        res = self._session.run(out, feeds)

        if log:
            i = kwargs['global_step']
            self._writer.add_summary(res[-1], global_step=i)
            res = res[: -1]

        return res


def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update


def entropy(x):  # the real concave entropy function
    x_move_reg = tf.clip_by_value((x + 1) / 2, 0.0001, 0.9999)
    pen = x_move_reg * tf.log(x_move_reg) + (1 - x_move_reg) * tf.log(1 - x_move_reg)
    return -tf.reduce_sum(pen, 1)


