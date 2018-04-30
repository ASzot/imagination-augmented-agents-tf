import os
import gym
import time
import logging
import numpy as np
import tensorflow as tf
from common.minipacman import MiniPacman
from common.multiprocessing_env import SubprocVecEnv
import tensorflow.contrib.slim as slim
from tqdm import tqdm


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        nh, nw, nc = ob_space

        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                        inputs=X,
                                        num_outputs=16,
                                        kernel_size=[3,3],
                                        stride=[1,1],
                                        padding='VALID')
            conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                        inputs=conv1,
                                        num_outputs=16,
                                        kernel_size=[3,3],
                                        stride=[2,2],
                                        padding='VALID')
            h = slim.fully_connected(slim.flatten(conv2), 256, activation_fn=tf.nn.relu)
            with tf.variable_scope('pi'):
                pi = slim.fully_connected(h, nact,
                        activation_fn=None,
                        weights_initializer=tf.random_normal_initializer(0.01),
                        biases_initializer=None)

            with tf.variable_scope('v'):
                vf = slim.fully_connected(h, 1,
                        activation_fn=None,
                        weights_initializer=tf.random_normal_initializer(0.01),
                        biases_initializer=None)[:, 0]

        # Sample action. `pi` is like the logits
        u = tf.random_uniform(tf.shape(pi))
        self.a0 = tf.argmax(pi- tf.log(-tf.log(u)), axis=-1)

        # Get the negative log likelihood
        one_hot_actions = tf.one_hot(self.a0, pi.get_shape().as_list()[-1])
        self.neglogp0 = tf.nn.softmax_cross_entropy_with_logits(
            logits=pi,
            labels=one_hot_actions)

        self.X = X
        self.pi = pi
        self.vf = vf

    def step(self, sess, ob):
        a, v, neglogp = sess.run([self.a0, self.vf, self.neglogp0], {self.X:ob})
        return a, v, neglogp

    def value(self, sess, ob):
        return sess.run(self.vf, {self.X:ob})


class ActorCritic(object):
    def __init__(self, sess, policy, ob_space, ac_space, nenvs, nsteps,
        ent_coef, vf_coef, max_grad_norm, lr, alpha, epsilon, logs_path='./logs/'):

        self.sess = sess

        nact = ac_space.n
        nbatch = nenvs*nsteps

        self.actions = tf.placeholder(tf.int32, [nbatch])
        self.advantages = tf.placeholder(tf.float32, [nbatch])
        self.rewards = tf.placeholder(tf.float32, [nbatch])

        self.step_model = policy(self.sess, ob_space, ac_space, nenvs, 1, reuse=False)
        self.train_model = policy(self.sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        # Negative log probability of actions
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_model.pi,
                labels=self.actions)

        # Policy gradient loss
        self.pg_loss = tf.reduce_mean(self.advantages * neglogpac)
        # Value function loss
        self.vf_loss = tf.reduce_mean(tf.square(tf.squeeze(self.train_model.vf) - self.rewards) / 2.0)
        self.entropy = tf.reduce_mean(cat_entropy(self.train_model.pi))
        self.loss = self.pg_loss - (self.entropy * ent_coef) + (self.vf_loss * vf_coef)

        with tf.variable_scope('model'):
            params = tf.trainable_variables()

        grads = tf.gradients(self.loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        self.opt = trainer.apply_gradients(grads)
        self.saver = tf.train.Saver(max_to_keep=15)

        # Tensorboard
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Policy gradient loss', self.pg_loss)
        tf.summary.scalar('Value function loss', self.vf_loss)

        self.summary_op = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(logs_path, graph=self.sess.graph)


    def train(self, obs, rewards, masks, actions, values, step):
        advs = rewards - values

        loss, policy_loss, value_loss, policy_entropy, _, summary = self.sess.run(
            [self.loss, self.pg_loss, self.vf_loss, self.entropy, self.opt,
                self.summary_op],
            feed_dict= {self.train_model.X: obs, self.actions: actions,
                self.advantages: advs, self.rewards: rewards}
        )

        self.writer.add_summary(summary, step)

        return loss, policy_loss, value_loss, policy_entropy

    def act(self, obs):
        return self.step_model.step(self.sess, obs)

    def critique(self, obs):
        return self.step_model.value(self.sess, obs)

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path + '/' + name)

    def load(self, full_path):
        self.saver.restore(self.sess, full_path)


def get_actor_critic(sess, nenvs, nsteps, ob_space, ac_space):
    vf_coef=0.5
    ent_coef=0.01
    max_grad_norm=0.5
    lr=7e-4
    epsilon=1e-5
    alpha=0.99
    actor_critic = ActorCritic(sess, CnnPolicy, ob_space, ac_space, nenvs, nsteps,
            ent_coef, vf_coef, max_grad_norm, lr, alpha, epsilon)

    return actor_critic

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    nenvs = 16
    nsteps=5
    total_timesteps=int(1e6)
    gamma=0.99
    log_interval=100
    save_interval = 1e5
    save_name = 'model'
    save_path = 'weights'

    load_count = 100000
    load_path = 'weights/model_%i.ckpt' % load_count
    #load_path = None

    def make_env():
        def _thunk():
            env = MiniPacman('regular', 1000)
            return env

        return _thunk

    envs = [make_env() for i in range(nenvs)]
    envs = SubprocVecEnv(envs)

    ob_space = envs.observation_space.shape
    nw, nh, nc = ob_space
    ac_space = envs.action_space

    obs = envs.reset()

    with tf.Session() as sess:
        actor_critic = get_actor_critic(sess, nenvs, nsteps, ob_space, ac_space)
        if load_path is not None:
            actor_critic.load(load_path)
            print('Loaded a2c')

        sess.run(tf.global_variables_initializer())

        batch_ob_shape = (nenvs*nsteps, nw, nh, nc)

        dones = [False for _ in range(nenvs)]

        nbatch = nenvs * nsteps

        episode_rewards = np.zeros((nenvs, ))
        final_rewards   = np.zeros((nenvs, ))

        for update in tqdm(range(load_count + 1, total_timesteps)):
            # mb stands for mini batch
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
            for n in range(nsteps):
                actions, values, _ = actor_critic.act(obs)

                mb_obs.append(np.copy(obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(dones)

                obs, rewards, dones, _ = envs.step(actions)

                episode_rewards += rewards
                masks = 1 - np.array(dones)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                mb_rewards.append(rewards)

            mb_dones.append(dones)

            #batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(batch_ob_shape)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
            mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
            mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]

            last_values = actor_critic.critique(obs).tolist()

            #discount/bootstrap off value fn
            for n, (rewards, d, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                d = d.tolist()
                if d[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], d+[0], gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, d, gamma)
                mb_rewards[n] = rewards

            mb_rewards = mb_rewards.flatten()
            mb_actions = mb_actions.flatten()
            mb_values = mb_values.flatten()
            mb_masks = mb_masks.flatten()

            loss, policy_loss, value_loss, policy_entropy = actor_critic.train(mb_obs,
                    mb_rewards, mb_masks, mb_actions, mb_values, update)

            if update % log_interval == 0 or update == 1:
                print('%i): %.4f, %.4f, %.4f' % (update, policy_loss, value_loss, policy_entropy))
                print(final_rewards.mean())

            if update % save_interval == 0 or update == 1:
                actor_critic.save(save_path, save_name + '_' + str(update) + '.ckpt')

        actor_critic.save(save_path, save_name + '_done.ckpt')

