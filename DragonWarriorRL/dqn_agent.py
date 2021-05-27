import time
import random

import keras.models
import numpy as np
from collections import deque
import tensorflow as tf
from matplotlib import pyplot as plt

# todo migrate to Tensorflow 2.0

class DQNAgent:
    '''DQN agent'''
    def __init__(self, states, actions, max_memory, double_q):
        self.states = states
        self.actions = actions
        self.session = tf.compat.v1.Session()
        self.build_model()
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10)
        self.saver.restore(self.session, './models/model')
        print('Model restored')
        self.session.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        self.memory = deque(maxlen=max_memory)
        self.eps = 1
        self.eps_decay = 0.99999975
        self.eps_min = 0.1
        self.gamma = 0.90
        self.batch_size = 32
        self.burnin = 100000000
        self.copy = 10000
        self.step = 0
        self.learn_each = 3
        self.learn_step = 0
        self.save_each = 40000
        self.double_q = double_q

    def build_model(self):
        '''Model builder function'''
        tf.compat.v1.disable_eager_execution()
        self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ) + self.states, name='input')
        self.q_true = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='labels')
        self.a_true = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.reward = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name='reward')
        self.input_float = tf.cast(self.input, dtype=tf.float32) / 255.
        # Online network
        with tf.compat.v1.variable_scope('online'):
            self.conv_1 = tf.compat.v1.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4,
                                           activation=tf.nn.relu)
            self.conv_2 = tf.compat.v1.layers.conv2d(inputs=self.conv_1, filters=64, kernel_size=4, strides=2,
                                           activation=tf.nn.relu)
            self.conv_3 = tf.compat.v1.layers.conv2d(inputs=self.conv_2, filters=64, kernel_size=3, strides=1,
                                           activation=tf.nn.relu)
            self.flatten = tf.compat.v1.layers.flatten(inputs=self.conv_3)
            # todo add in map_id, map_position, magic_keys, hero stats, gold, HP
            self.dense = tf.compat.v1.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu)
            self.output = tf.compat.v1.layers.dense(inputs=self.dense, units=self.actions, name='output')
        # Target network
        with tf.compat.v1.variable_scope('target'):
            self.conv_1_target = tf.compat.v1.layers.conv2d(inputs=self.input_float, filters=32, kernel_size=8, strides=4,
                                                  activation=tf.nn.relu)
            self.conv_2_target = tf.compat.v1.layers.conv2d(inputs=self.conv_1_target, filters=64, kernel_size=4, strides=2,
                                                  activation=tf.nn.relu)
            self.conv_3_target = tf.compat.v1.layers.conv2d(inputs=self.conv_2_target, filters=64, kernel_size=3, strides=1,
                                                  activation=tf.nn.relu)
            self.flatten_target = tf.compat.v1.layers.flatten(inputs=self.conv_3_target)
            # todo add in map_id, map_position, magic_keys, hero stats, gold, HP
            self.dense_target = tf.compat.v1.layers.dense(inputs=self.flatten_target, units=512, activation=tf.nn.relu)
            self.output_target = tf.stop_gradient(
                tf.compat.v1.layers.dense(inputs=self.dense_target, units=self.actions, name='output_target'))
        # Optimizer
        self.action = tf.argmax(input=self.output, axis=1)
        self.q_pred = tf.gather_nd(params=self.output,
                                   indices=tf.stack([tf.range(tf.shape(self.a_true)[0]), self.a_true], axis=1))
        self.loss = tf.compat.v1.losses.huber_loss(labels=self.q_true, predictions=self.q_pred)
        self.train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00025).minimize(self.loss)
        # Summaries
        self.summaries = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar('reward', self.reward),
            tf.compat.v1.summary.scalar('loss', self.loss),
            tf.compat.v1.summary.scalar('max_q', tf.reduce_max(self.output))
        ])
        self.writer = tf.compat.v1.summary.FileWriter(logdir='./logs', graph=self.session.graph)

    def copy_model(self):
        """ Copy weights to target network """
        self.session.run([tf.compat.v1.assign(new, old) for (new, old) in
                          zip(tf.compat.v1.trainable_variables('target'), tf.compat.v1.trainable_variables('online'))])

    def save_model(self):
        """ Saves current model to disk """
        # self.saver.save(sess=self.session, save_path='./models/model', global_step=self.step)
        self.saver.save(sess=self.session, save_path='./models/model')


    def add(self, experience):
        """ Add observation to experience """
        self.memory.append(experience)

    def predict(self, model, state):
        """ Prediction """
        if model == 'online':
            return self.session.run(fetches=self.output, feed_dict={self.input: np.array(state)})
        if model == 'target':
            return self.session.run(fetches=self.output_target, feed_dict={self.input: np.array(state)})

    def run(self, state):
        """ Perform action """
        if np.random.rand() < self.eps:
            # Random action
            action = np.random.randint(low=0, high=self.actions)
        else:
            # Policy action
            q = self.predict('online', np.expand_dims(state, 0))
            action = np.argmax(q)
        # Decrease eps
        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        # Increment step
        self.step += 1
        return action

    def learn(self):
        """ Gradient descent """
        # Sync target network
        if self.step % self.copy == 0:
            self.copy_model()
        # Checkpoint model
        if self.step % self.save_each == 0:
            self.save_model()
        # Break if burn-in
        if self.step < self.burnin:
            return
        # Break if no training
        if self.learn_step < self.learn_each:
            self.learn_step += 1
            return
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))
        # Get next q values from target network
        next_q = self.predict('target', next_state)
        # Calculate discounted future reward
        if self.double_q:
            q = self.predict('online', next_state)
            a = np.argmax(q, axis=1)
            target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
        else:
            target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1)
        # Update model
        summary, _ = self.session.run(fetches=[self.summaries, self.train],
                                      feed_dict={self.input: state,
                                                 self.q_true: np.array(target_q),
                                                 self.a_true: np.array(action),
                                                 self.reward: np.mean(reward)})
        # Reset learn step
        self.learn_step = 0
        # Write
        self.writer.add_summary(summary, self.step)