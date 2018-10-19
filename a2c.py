import os
import numpy as np
import tensorflow as tf

# TUNABLE HYPERPARAMETERS FOR A2C TRAINING
VF_COEFF=0.5
ENTROPY_COEFF=0.01
MAX_GRAD_NORM=0.5
LR=7e-4
EPSILON=1e-5
ALPHA=0.99

def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

# Basic baseline policy
class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        nw, nh, nc = ob_space

        nact = ac_space.n
        X = tf.placeholder(tf.float32, [None, nw, nh, nc]) #obs
        with tf.variable_scope("model", reuse=reuse):
            conv1 = tf.layers.conv2d(activation=tf.nn.relu,
                                        inputs=X,
                                        filters=16,
                                        kernel_size=[3,3],
                                        strides=[1,1],
                                        padding='VALID')
            conv2 = tf.layers.conv2d(activation=tf.nn.relu,
                                        inputs=conv1,
                                        filters=16,
                                        kernel_size=[3,3],
                                        strides=[2,2],
                                        padding='VALID')
            h = tf.layers.dense(tf.layers.flatten(conv2), 256, activation=tf.nn.relu)
            with tf.variable_scope('pi'):
                pi = tf.layers.dense(h, nact,
                        activation=None,
                        kernel_initializer=tf.random_normal_initializer(0.01),
                        bias_initializer=None)

            with tf.variable_scope('v'):
                vf = tf.layers.dense(h, 1,
                        activation=None,
                        kernel_initializer=tf.random_normal_initializer(0.01),
                        bias_initializer=None)[:, 0]

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
        v = sess.run(self.vf, {self.X:ob})
        return v

    # Next two methods are required when we will have to generate the imaginations later in the I2A
    # code.
    def transform_input(self, X, sess):
        return [X]

    def get_inputs(self):
        return [self.X]


# generic graph for a2c.
class ActorCritic(object):
    def __init__(self, sess, policy, ob_space, ac_space, nenvs, nsteps, should_summary):
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
        self.loss = self.pg_loss - (self.entropy * ENTROPY_COEFF) + (self.vf_loss * VF_COEFF)

        with tf.variable_scope('model'):
            params = tf.trainable_variables()

        grads = tf.gradients(self.loss, params)
        if MAX_GRAD_NORM is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=ALPHA,
                epsilon=EPSILON)
        self.opt = trainer.apply_gradients(grads)

        # Tensorboard
        if should_summary:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Policy gradient loss', self.pg_loss)
            tf.summary.scalar('Value function loss', self.vf_loss)

        name_scope = tf.contrib.framework.get_name_scope()

        # Used if we are loading in a scope different than what we saved in.
        def fix_tf_name(name, name_scope=None):
            if name_scope is not None:
                name = name[len(name_scope) + 1:]
            return name.split(':')[0]

        if len(name_scope) != 0:
            params = {fix_tf_name(v.name, name_scope): v for v in params}
        else:
            params = {fix_tf_name(v.name): v for v in params}

        self.saver = tf.train.Saver(params, max_to_keep=15)


    # generic training code for one iteration.
    def train(self, obs, rewards, masks, actions, values, step, summary_op=None):
        advs = rewards - values

        feed_dict = {
            self.actions: actions,
            self.advantages: advs,
            self.rewards: rewards
        }

        inputs = self.train_model.get_inputs()
        mapped_input = self.train_model.transform_input(obs, self.sess)
        for transformed_input, inp in zip(mapped_input, inputs):
            feed_dict[inp] = transformed_input

        ret_vals = [
                self.loss,
                self.pg_loss,
                self.vf_loss,
                self.entropy,
                self.opt,
            ]

        if summary_op is not None:
            ret_vals.append(summary_op)

        results = self.sess.run(
            ret_vals,
            feed_dict = feed_dict
        )

        return results


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


def get_actor_critic(sess, nenvs, nsteps, ob_space, ac_space,
        policy, should_summary=True):

    actor_critic = ActorCritic(sess, policy, ob_space, ac_space, nenvs, nsteps, should_summary)

    return actor_critic

