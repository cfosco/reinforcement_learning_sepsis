# -*- coding: utf-8 -*-
"""
OpenAI winning ActorCritic implementation for solving MountainCarContinuous
"""

import tensorflow as tf
import numpy as np
import os
import gym
import time
import sklearn
import itertools
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func


env = gym.envs.make("MountainCarContinuous-v0")
video_dir = os.path.abspath("./videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
#env = gym.wrappers.Monitor(env, video_dir, force=True)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def process_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]


class PolicyEstimator:
    def __init__(self, env, lamb=1e-5, learning_rate=0.01, scope="policy_estimator_openAI"):
        self.env = env
        self.lamb = lamb
        self.learning_rate = learning_rate

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [400], name="state")

        self.mu = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.mu = tf.squeeze(self.mu)

        self.sigma = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.norm_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])

    def _build_train_op(self):
        self.action_train = tf.placeholder(tf.float32, name="action_train")
        self.advantage_train = tf.placeholder(tf.float32, name="advantage_train")

        self.loss = -tf.log(
            self.norm_dist.prob(self.action_train) + 1e-5) * self.advantage_train - self.lamb * self.norm_dist.entropy()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
    def get_q_function_params(self, state, sess):
        feed_dict = {self.state: process_state(state)}
        return sess.run([self.mu, self.sigma], feed_dict=feed_dict) 

    def predict(self, state, sess):
        feed_dict = {self.state: process_state(state)}
        return sess.run(self.action, feed_dict=feed_dict)

    def update(self, state, action, advantage, sess):
        feed_dict = {
            self.state: process_state(state),
            self.action_train: action,
            self.advantage_train: advantage
        }
        sess.run([self.train_op], feed_dict=feed_dict)


class ValueEstimator:
    def __init__(self, env, learning_rate=0.01, scope="value_estimator"):
        self.env = env
        self.learning_rate = learning_rate

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [400], name="state")

        self.value = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.value = tf.squeeze(self.value)

    def _build_train_op(self):
        self.target = tf.placeholder(tf.float32, name="target")
        self.loss = tf.reduce_mean(tf.squared_difference(self.value, self.target))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: process_state(state)})

    def update(self, state, target, sess):
        feed_dict = {
            self.state: process_state(state),
            self.target: target
        }
        sess.run([self.train_op], feed_dict=feed_dict)


@exec_time
def actor_critic(episodes=100, gamma=0.95, display=False, lamb=1e-5, policy_lr=0.001, value_lr=0.1):
    tf.reset_default_graph()
    policy_estimator = PolicyEstimator(env, lamb=lamb, learning_rate=policy_lr)
    value_estimator = ValueEstimator(env, learning_rate=value_lr)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    stats = []
    histories = {}
    for i_episode in range(episodes):
        histories[i_episode]=[]
        state = env.reset()
        reward_total = 0
        for t in itertools.count():
            action = policy_estimator.predict(state, sess)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            histories[i_episode].append([state, action, reward, next_state])
            if display:
                env.render()

            target = reward + gamma * value_estimator.predict(next_state, sess)
            td_error = target - value_estimator.predict(state, sess)

            if i_episode<20:
                policy_estimator.update(state, action, advantage=td_error, sess=sess)
                value_estimator.update(state, target, sess=sess)

            if done:
                break
            state = next_state
        stats.append(reward_total)
        if np.mean(stats[-100:]) > 90 and len(stats) >= 101:
            print(np.mean(stats[-100:]))
            print("Solved")
        print("Episode: {}, reward: {}, length: {}".format(i_episode, reward_total, len(histories[i_episode])))
    return np.mean(stats[-100:]), histories, policy_estimator, value_estimator, sess


if __name__ == "__main__":
    policy_lr, value_lr, lamb, gamma = [0.0001, 0.00046415888336127773, 2.782559402207126e-05, 0.98999999999999999]
    loss = actor_critic(episodes=1000, gamma=gamma, display=False, lamb=lamb, policy_lr=policy_lr, value_lr=value_lr)
    print(-loss)
    env.close()