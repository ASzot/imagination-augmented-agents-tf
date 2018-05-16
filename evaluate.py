import numpy as np
import os
from a2c import get_actor_critic, CnnPolicy
from common.minipacman import MiniPacman
import tensorflow as tf
from i2a import I2aPolicy

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

nenvs = 1
nsteps=5

done = False
env = MiniPacman('regular', 1000)
ob_space = env.observation_space.shape
nw, nh, nc = ob_space
ac_space = env.action_space

states = env.reset()


with tf.Session() as sess:
    actor_critic = get_actor_critic(sess, nenvs, nsteps, ob_space, ac_space,
            CnnPolicy, False)
    actor_critic.load('./weights/model_100000.ckpt')

    total_reward = 0

    while not done:
        states = np.expand_dims(states, 0)
        actions, values, _ = actor_critic.act(states)

        states, reward, done, _ = env.step(actions[0])

        total_reward += reward

    print('total reward', total_reward)
