# To train the environment model. See paper appendix for implementation
# details.

import os
import tensorflow as tf
from common.minipacman import MiniPacman
from a2c import get_actor_critic, CnnPolicy
from common.multiprocessing_env import SubprocVecEnv
import numpy as np

from tqdm import tqdm

from common.pacman_util import num_pixels, mode_rewards, pix_to_target, rewards_to_target

# How many iterations we are training the environment model for.
NUM_UPDATES = 5000

LOG_INTERVAL = 100

N_ENVS = 16
N_STEPS = 5

# This can be anything from "regular" "avoid" "hunt" "ambush" "rush" each
# resulting in a different reward function giving the agent different behavior.
REWARD_MODE = 'regular'

# Replace this with the location of your own weights.
A2C_WEIGHTS = 'weights/a2c_200000.ckpt'

def pool_inject(X, batch_size, depth, width, height):
    m = tf.layers.max_pooling2d(X, pool_size=(width, height), strides=(width, height))
    tiled = tf.tile(m, (1, width, height, 1))
    return tf.concat([tiled, X], axis=-1)

def basic_block(X, batch_size, depth, width, height, n1, n2, n3):
    with tf.variable_scope('pool_inject'):
        p = pool_inject(X, batch_size, depth, width, height)

    with tf.variable_scope('part_1_block'):
        # Padding was 6 here
        p_padded = tf.pad(p, [[0, 0], [6, 6], [6, 6], [0, 0]])
        p_1_c1 = tf.layers.conv2d(p_padded, n1, kernel_size=1,
                strides=2, padding='valid', activation=tf.nn.relu)

        # Padding was 5, 6
        p_1_c1 = tf.pad(p_1_c1, [[0,0], [5, 5], [6, 6], [0, 0]])
        p_1_c2 = tf.layers.conv2d(p_1_c1, n1, kernel_size=10, strides=1,
                padding='valid', activation=tf.nn.relu)

    with tf.variable_scope('part_2_block'):
        p_2_c1 = tf.layers.conv2d(p, n2, kernel_size=1,
                activation=tf.nn.relu)

        p_2_c1 = tf.pad(p_2_c1, [[0,0],[1,1],[1,1],[0,0]])
        p_2_c2 = tf.layers.conv2d(p_2_c1, n2, kernel_size=3, strides=1,
                padding='valid', activation=tf.nn.relu)

    with tf.variable_scope('combine_parts'):
        combined = tf.concat([p_1_c2, p_2_c2], axis=-1)

        c = tf.layers.conv2d(combined, n3, kernel_size=1,
                activation=tf.nn.relu)

    return tf.concat([c, X], axis=-1)


def create_env_model(obs_shape, num_actions, num_pixels, num_rewards,
        should_summary=True, reward_coeff=0.1):
    width = obs_shape[0]
    height = obs_shape[1]
    depth = obs_shape[2]

    states = tf.placeholder(tf.float32, [None, width, height, depth])

    onehot_actions = tf.placeholder(tf.float32, [None, width,
        height, num_actions])

    batch_size = tf.shape(states)[0]

    target_states = tf.placeholder(tf.uint8, [None])
    target_rewards = tf.placeholder(tf.uint8, [None])

    inputs = tf.concat([states, onehot_actions], axis=-1)

    with tf.variable_scope('pre_conv'):
        c = tf.layers.conv2d(inputs, 64, kernel_size=1, activation=tf.nn.relu)

    with tf.variable_scope('basic_block_1'):
        bb1 = basic_block(c, batch_size, 64, width, height, 16, 32, 64)

    with tf.variable_scope('basic_block_2'):
        bb2 = basic_block(bb1, batch_size, 128, width, height, 16, 32, 64)


    with tf.variable_scope('image_conver'):
        image = tf.layers.conv2d(bb2, 256, kernel_size=1, activation=tf.nn.relu)
        image = tf.reshape(image, [batch_size * width * height, 256])
        image = tf.layers.dense(image, num_pixels)

    with tf.variable_scope('reward'):
        reward = tf.layers.conv2d(bb2, 64, kernel_size=1,
                activation=tf.nn.relu)

        reward = tf.layers.conv2d(reward, 64, kernel_size=1,
                activation=tf.nn.relu)

        reward = tf.reshape(reward, [batch_size, width * height * 64])

        reward = tf.layers.dense(reward, num_rewards)

    target_states_one_hot = tf.one_hot(target_states, depth=num_pixels)
    image_loss = tf.losses.softmax_cross_entropy(target_states_one_hot, image)

    target_reward_one_hot = tf.one_hot(target_rewards, depth=num_rewards)
    reward_loss = tf.losses.softmax_cross_entropy(target_reward_one_hot, reward)

    loss = image_loss + (reward_coeff * reward_loss)

    opt = tf.train.AdamOptimizer().minimize(loss)

    # Tensorboard
    if should_summary:
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Reward Loss', reward_loss)
        tf.summary.scalar('Image Loss', image_loss)

    return EnvModelData(image, reward, states, onehot_actions, loss,
            reward_loss, image_loss, target_states, target_rewards, opt)


def make_env():
    def _thunk():
        env = MiniPacman(REWARD_MODE, 1000)
        return env

    return _thunk

def play_games(actor_critic, envs, frames):
    states = envs.reset()

    for frame_idx in range(frames):
        actions, _, _ = actor_critic.act(states)

        next_states, rewards, dones, _ = envs.step(actions)

        yield frame_idx, states, actions, rewards, next_states, dones

        states = next_states


class EnvModelData(object):
    def __init__(self, imag_state, imag_reward, input_states, input_actions,
            loss, reward_loss, image_loss, target_states, target_rewards, opt):
        self.imag_state       = imag_state
        self.imag_reward      = imag_reward
        self.input_states     = input_states
        self.input_actions    = input_actions

        self.loss             = loss
        self.reward_loss      = reward_loss
        self.image_loss       = image_loss
        self.target_states    = target_states
        self.target_rewards   = target_rewards
        self.opt              = opt


if __name__ == '__main__':
    envs = [make_env() for i in range(N_ENVS)]
    envs = SubprocVecEnv(envs)

    ob_space = envs.observation_space.shape
    ac_space = envs.action_space
    num_actions = envs.action_space.n

    with tf.Session() as sess:
        actor_critic = get_actor_critic(sess, N_ENVS, N_STEPS, ob_space, ac_space, CnnPolicy, should_summary=False)
        actor_critic.load(A2C_WEIGHTS)

        with tf.variable_scope('env_model'):
            env_model = create_env_model(ob_space, num_actions, num_pixels,
                    len(mode_rewards[REWARD_MODE]))

        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        losses = []
        all_rewards = []

        width = ob_space[0]
        height = ob_space[1]
        depth = ob_space[2]

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        saver = tf.train.Saver(var_list=save_vars)

        writer = tf.summary.FileWriter('./env_logs', graph=sess.graph)

        for frame_idx, states, actions, rewards, next_states, dones in tqdm(play_games(actor_critic, envs, NUM_UPDATES), total=NUM_UPDATES):
            target_state = pix_to_target(next_states)
            target_reward = rewards_to_target(REWARD_MODE, rewards)

            onehot_actions = np.zeros((N_ENVS, num_actions, width, height))
            onehot_actions[range(N_ENVS), actions] = 1
            # Change so actions are the 'depth of the image' as tf expects
            onehot_actions = onehot_actions.transpose(0, 2, 3, 1)

            s, r, l, reward_loss, image_loss, summary, _ = sess.run([
                env_model.imag_state,
                env_model.imag_reward,
                env_model.loss,
                env_model.reward_loss,
                env_model.image_loss,
                summary_op,
                env_model.opt], feed_dict={
                    env_model.input_states: states,
                    env_model.input_actions: onehot_actions,
                    env_model.target_states: target_state,
                    env_model.target_rewards: target_reward
                })

            if frame_idx % LOG_INTERVAL == 0:
                print('%i) %.5f, %.5f, %.5f' % (frame_idx, l, reward_loss, image_loss))
            writer.add_summary(summary, frame_idx)

        saver.save(sess, 'weights/env_model.ckpt')
        print('Environment model saved!')

