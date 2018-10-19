import os
import gym
import time
import logging
import numpy as np
import tensorflow as tf
from common.minipacman import MiniPacman
from common.multiprocessing_env import SubprocVecEnv
from tqdm import tqdm

from env_model import create_env_model
from a2c import get_actor_critic, CnnPolicy
from common.pacman_util import num_pixels, mode_rewards, pix_to_target, rewards_to_target, mode_rewards, target_to_pix


# Hyperparameter of how far ahead in the future the agent "imagines"
# Currently this is specifying one frame in the future.
NUM_ROLLOUTS = 1

# Hidden size in RNN imagination encoder.
HIDDEN_SIZE = 256

N_STEPS = 5

# This can be anything from "regular" "avoid" "hunt" "ambush" "rush" each
# resulting in a different reward function giving the agent different behavior.
REWARD_MODE = 'regular'

# Replace this with the name of the weights you want to load to train I2A
A2C_MODEL_PATH = 'weights/a2c_200000.ckpt'
ENV_MODEL_PATH = 'weights/env_model.ckpt'

# Softmax function for numpy taken from
# https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def convert_target_to_real(batch_size, nw, nh, nc, imagined_state, imagined_reward):
    imagined_state = softmax(imagined_state, axis=1)
    imagined_state = np.argmax(imagined_state, axis=1)
    imagined_state = target_to_pix(imagined_state)
    imagined_state = imagined_state.reshape((batch_size, nw, nh,
        nc))

    imagined_reward = softmax(imagined_reward, axis=1)
    imagined_reward = np.argmax(imagined_reward, axis=1)

    return imagined_state, imagined_reward


"""
Used to generate rollouts of imagined states.
"""
class ImaginationCore(object):
    def __init__(self, num_rollouts, num_actions, num_rewards,
            ob_space, actor_critic, env_model):

        self.num_rollouts = num_rollouts
        self.num_actions  = num_actions
        self.num_rewards  = num_rewards
        self.ob_space     = ob_space
        self.actor_critic = actor_critic
        self.env_model    = env_model


    def imagine(self, state, sess):
        nw, nh, nc = self.ob_space

        batch_size = state.shape[0]

        state = np.tile(state, [self.num_actions, 1, 1, 1, 1])
        state = state.reshape(-1, nw, nh, nc)

        action = np.array([[[i] for i in range(self.num_actions)] for j in
            range(batch_size)])

        action = action.reshape((-1,))

        rollout_batch_size = batch_size * self.num_actions

        rollout_states = []
        rollout_rewards = []

        for step in range(self.num_rollouts):
            onehot_action = np.zeros((rollout_batch_size, self.num_actions, nw, nh))
            onehot_action[range(rollout_batch_size), action] = 1
            onehot_action = np.transpose(onehot_action, (0, 2, 3, 1))

            imagined_state, imagined_reward = sess.run(
                    [self.env_model.imag_state, self.env_model.imag_reward],
                    feed_dict={
                        self.env_model.input_states: state,
                        self.env_model.input_actions: onehot_action,
                })

            imagined_state, imagined_reward = convert_target_to_real(rollout_batch_size, nw, nh, nc, imagined_state, imagined_reward)

            onehot_reward = np.zeros((rollout_batch_size, self.num_rewards))
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state)
            rollout_rewards.append(onehot_reward)

            state = imagined_state
            action, _, _ = self.actor_critic.act(state)

        return np.array(rollout_states), np.array(rollout_rewards)

# So the model is not loaded twice.
g_actor_critic = None
def get_cache_loaded_a2c(sess, nenvs, nsteps, ob_space, ac_space):
    global g_actor_critic
    if g_actor_critic is None:
        with tf.variable_scope('actor'):
            g_actor_critic = get_actor_critic(sess, nenvs, nsteps, ob_space,
                    ac_space, CnnPolicy, should_summary=False)
        g_actor_critic.load(A2C_MODEL_PATH)

        print('Actor restored!')
    return g_actor_critic


# So the model is not loaded twice.
g_env_model = None
def get_cache_loaded_env_model(sess, nenvs, ob_space, num_actions):
    global g_env_model
    if g_env_model is None:
        with tf.variable_scope('env_model'):
            g_env_model = create_env_model(ob_space, num_actions, num_pixels,
                    len(mode_rewards[REWARD_MODE]), should_summary=False)

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        loader = tf.train.Saver(var_list=save_vars)
        loader.restore(sess, ENV_MODEL_PATH)

        print('Env model restored!')

    return g_env_model


class I2aPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        num_rewards = len(mode_rewards[REWARD_MODE])
        num_actions = ac_space.n
        width, height, depth = ob_space

        actor_critic = get_cache_loaded_a2c(sess, nbatch, N_STEPS, ob_space, ac_space)
        env_model = get_cache_loaded_env_model(sess, nbatch, ob_space, num_actions)

        self.imagination = ImaginationCore(NUM_ROLLOUTS, num_actions, num_rewards,
                ob_space, actor_critic, env_model)

        with tf.variable_scope('model', reuse=reuse):
            # Model based path.
            self.imagined_state = tf.placeholder(tf.float32, [None, None, width, height, depth])
            self.imagined_reward = tf.placeholder(tf.float32, [None, None, num_rewards])

            num_steps = tf.shape(self.imagined_state)[0]
            batch_size = tf.shape(self.imagined_state)[1]

            hidden_state = self.get_encoder(self.imagined_state, self.imagined_reward,
                    num_steps, batch_size, width, height, depth, HIDDEN_SIZE)

            # Model free path.
            self.state = tf.placeholder(tf.float32, [None, width, height,
                depth])

            state_batch_size = tf.shape(self.state)[0]

            c1 = tf.layers.conv2d(self.state, 16, kernel_size=3,
                    strides=1, padding='valid', activation=tf.nn.relu)
            c2 = tf.layers.conv2d(c1, 16, kernel_size=3,
                    strides=2, padding='valid', activation=tf.nn.relu)

            features = tf.reshape(c2, [state_batch_size, 6 * 8 * 16])

            self.features = features

            hidden_state = tf.reshape(hidden_state, [state_batch_size, 80 * 256
                // 16])

            # Combine both paths
            x = tf.concat([features, hidden_state], axis=1)
            x = tf.layers.dense(x, 256, activation=tf.nn.relu)

            self.pi = tf.layers.dense(x, num_actions)
            self.vf = tf.layers.dense(x, 1)[:, 0]

        # Sample action. `pi` is like the logits
        u = tf.random_uniform(tf.shape(self.pi))
        self.a0 = tf.argmax(self.pi - tf.log(-tf.log(u)), axis=-1)

        # Get the negative log likelihood
        one_hot_actions = tf.one_hot(self.a0, self.pi.get_shape().as_list()[-1])
        self.neglogp0 = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.pi,
            labels=one_hot_actions)


    def get_encoder(self, state, reward, num_steps, batch_size, width, height, depth, hidden_size):
        state = tf.reshape(state, [num_steps * batch_size, width, height,
            depth])

        c1 = tf.layers.conv2d(state, 16, kernel_size=3, strides=1,
                padding='valid', activation=tf.nn.relu)
        features = tf.layers.conv2d(c1, 16, kernel_size=3, strides=2,
                padding='valid', activation=tf.nn.relu)

        features = tf.reshape(features, [num_steps, batch_size, 6 * 8 * 16])

        rnn_input = tf.concat([features, reward], 2)

        cell = tf.contrib.rnn.GRUCell(hidden_size)
        _, internal_state = tf.nn.dynamic_rnn(cell, rnn_input, time_major=True, dtype=tf.float32)

        return internal_state


    def step(self, sess, ob):
        imagined_state, imagined_reward, ob = self.transform_input(ob, sess)

        a, v, neglogp = sess.run([
                self.a0,
                self.vf,
                self.neglogp0
            ],
            {
                self.imagined_state: imagined_state,
                self.imagined_reward: imagined_reward,
                self.state: ob
        })
        return a, v, neglogp


    def value(self, sess, ob):
        imagined_state, imagined_reward, ob = self.transform_input(ob, sess)

        v = sess.run(self.vf, {
            self.imagined_state: imagined_state,
            self.imagined_reward: imagined_reward,
            self.state: ob
        })
        return v

    # Add the imagined states to the default input.
    def get_inputs(self):
        return [self.imagined_state, self.imagined_reward, self.state]

    def transform_input(self, X, sess):
        imagined_state, imagined_reward = self.imagination.imagine(X, sess)
        return [imagined_state, imagined_reward, X]

