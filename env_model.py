import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from common.minipacman import MiniPacman
from a2c import get_actor_critic
from common.multiprocessing_env import SubprocVecEnv
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#7 different pixels in MiniPacman
pixels = (
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
)
pixel_to_categorical = {pix:i for i, pix in enumerate(pixels)}
num_pixels = len(pixels)

#For each mode in MiniPacman there are different rewards
mode_rewards = {
    "regular": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "avoid":   [0.1, -0.1, -5, -10, -20],
    "hunt":    [0, 1, 10, -20],
    "ambush":  [0, -0.1, 10, -20],
    "rush":    [0, -0.1, 9.9]
}
reward_to_categorical = {mode: {reward:i for i, reward in enumerate(mode_rewards[mode])} for mode in mode_rewards.keys()}

def pix_to_target(next_states):
    target = []
    for pixel in next_states.transpose(0, 2, 3, 1).reshape(-1, 3):
        target.append(pixel_to_categorical[tuple([np.ceil(pixel[0]), np.ceil(pixel[1]), np.ceil(pixel[2])])])
    return target

def target_to_pix(imagined_states):
    pixels = []
    to_pixel = {value: key for key, value in pixel_to_categorical.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))
    return np.array(pixels)

def rewards_to_target(mode, rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_categorical[mode][reward])
    return target

def pool_inject(X, batch_size, depth, width, height):
    m = slim.max_pool2d(X, kernel_size=(width, height))
    tiled = tf.tile(m, (1, width, height, 1))
    return tf.concat([tiled, X], axis=-1)


def basic_block(X, batch_size, depth, width, height, n1, n2, n3):
    with tf.variable_scope('pool_inject'):
        p = pool_inject(X, batch_size, depth, width, height)

    with tf.variable_scope('part_1_block'):
        # Padding was 6 here
        p_padded = tf.pad(p, [[0, 0], [6, 6], [6, 6], [0, 0]])
        p_1_c1 = slim.conv2d(p_padded, n1, kernel_size=1,
                stride=2, padding='valid', activation_fn=tf.nn.relu)

        # Padding was 5, 6
        p_1_c1 = tf.pad(p_1_c1, [[0,0], [5, 5], [6, 6], [0, 0]])
        p_1_c2 = slim.conv2d(p_1_c1, n1, kernel_size=10, stride=1,
                padding='valid', activation_fn=tf.nn.relu)

    with tf.variable_scope('part_2_block'):
        p_2_c1 = slim.conv2d(p, n2, kernel_size=1,
                activation_fn=tf.nn.relu)

        p_2_c1 = tf.pad(p_2_c1, [[0,0],[1,1],[1,1],[0,0]])
        p_2_c2 = slim.conv2d(p_2_c1, n2, kernel_size=3, stride=1,
                padding='valid', activation_fn=tf.nn.relu)

    with tf.variable_scope('combine_parts'):
        combined = tf.concat([p_1_c2, p_2_c2], axis=-1)

        c = slim.conv2d(combined, n3, kernel_size=1,
                activation_fn=tf.nn.relu)

    return tf.concat([c, X], axis=-1)


def env_model(obs_shape, batch_size, num_pixels, num_rewards, reward_coeff=0.1):
    width = obs_shape[0]
    height = obs_shape[1]
    depth = obs_shape[2]

    states = tf.placeholder(tf.float32, [None, width, height, depth])
    onehot_actions = tf.placeholder(tf.float32, [None, width,
        height, num_actions])

    target_states = tf.placeholder(tf.uint8, [batch_size * width * height])
    target_rewards = tf.placeholder(tf.uint8, [batch_size])

    #target_states = tf.placeholder(tf.float32, [None, width, height, depth])

    inputs = tf.concat([states, onehot_actions], axis=-1)

    with tf.variable_scope('pre_conv'):
        c = slim.conv2d(inputs, 64, kernel_size=1, activation_fn=tf.nn.relu)

    with tf.variable_scope('basic_block_1'):
        bb1 = basic_block(c, batch_size, 64, width, height, 16, 32, 64)

    with tf.variable_scope('basic_block_2'):
        bb2 = basic_block(bb1, batch_size, 128, width, height, 16, 32, 64)

    with tf.variable_scope('image_conver'):
        image = slim.conv2d(bb2, 256, kernel_size=1, activation_fn=tf.nn.relu)
        image = tf.reshape(image, [batch_size * width * height, 256])
        image = slim.fully_connected(image, num_pixels)

    with tf.variable_scope('reward'):
        reward = slim.conv2d(bb2, 64, kernel_size=1,
                activation_fn=tf.nn.relu)
        reward = slim.conv2d(reward, 64, kernel_size=1, activation_fn=tf.nn.relu)

        reward = tf.reshape(reward, [batch_size, width * height * 64])

        reward = slim.fully_connected(reward, num_rewards)

    target_states_one_hot = tf.one_hot(target_states, depth=num_pixels)
    image_loss = tf.losses.softmax_cross_entropy(target_states_one_hot, image)

    target_reward_one_hot = tf.one_hot(target_rewards, depth=num_rewards)
    reward_loss = tf.losses.softmax_cross_entropy(target_reward_one_hot, reward)

    loss = image_loss + (reward_coeff * reward_loss)

    opt = tf.train.AdamOptimizer().minimize(loss)

    return image, reward, states, onehot_actions, loss, target_states, target_rewards


nenvs = 16
nsteps = 5

def make_env():
    def _thunk():
        env = MiniPacman('regular', 1000)
        return env

    return _thunk

envs = [make_env() for i in range(nenvs)]
envs = SubprocVecEnv(envs)

env = MiniPacman('regular', 1000)
ob_space = env.observation_space.shape
ac_space = env.action_space
num_actions = envs.action_space.n

def play_games(envs, frames):
    states = envs.reset()

    for frame_idx in range(frames):
        tmp_states = states.transpose(0, 2, 1, 3)
        actions, _, _ = actor_critic.act(tmp_states)

        next_states, rewards, dones, _ = envs.step(actions)

        yield frame_idx, states, actions, rewards, next_states, dones

        states = next_states


with tf.Session() as sess:
    actor_critic = get_actor_critic(sess, nenvs, nsteps, [ob_space[1], ob_space[0],
        ob_space[2]], ac_space)
    actor_critic.load('weights/model_1.ckpt')

    imag_state, imag_reward, input_states, input_actions, loss, target_states, target_rewards = env_model(ob_space, nenvs, num_pixels, len(mode_rewards['regular']))
    sess.run(tf.global_variables_initializer())

    reward_coef = 0.1
    num_updates = 5000

    losses = []
    all_rewards = []

    width = ob_space[0]
    height = ob_space[1]
    depth = ob_space[2]

    for frame_idx, states, actions, rewards, next_states, dones in play_games(envs, num_updates):
        target_state = pix_to_target(next_states)
        target_reward = rewards_to_target('regular', rewards)

        onehot_actions = np.zeros((nenvs, num_actions, width, height))
        onehot_actions[range(nenvs), actions] = 1
        # Change so actions are the 'depth of the image' as tf expects
        onehot_actions = onehot_actions.transpose(0, 2, 3, 1)

        s, r, l = sess.run([imag_state, imag_reward, loss], feed_dict={
                input_states: states,
                input_actions: onehot_actions,
                target_states: target_state,
                target_rewards: target_reward
            })

        raise ValueError()




