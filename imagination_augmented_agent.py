import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from common.multiprocessing_env import SubprocVecEnv
from common.minipacman import MiniPacman
from common.environment_model import EnvModel
from common.actor_critic import OnPolicy, ActorCritic, RolloutStorage
import time

from common.rollout_encoder import RolloutEncoder
from common.imagination_core import ImaginationCore
from common.i2a import I2A

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

SHOULD_LOG = False

def plog(s, time=''):
    if SHOULD_LOG:
        print(s, time)


mode = "regular"
num_envs = 16

def make_env():
    def _thunk():
        env = MiniPacman(mode, 1000)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_shape = envs.observation_space.shape
num_actions = envs.action_space.n
num_rewards = len(task_rewards[mode])

full_rollout = True

# Get the env model which is trained to predict the next frame and the reward
# associated with the current frame
env_model     = EnvModel(envs.observation_space.shape, num_pixels, num_rewards)
env_model.load_state_dict(torch.load("env_model_" + mode))

distil_policy = ActorCritic(envs.observation_space.shape, envs.action_space.n)
distil_optimizer = optim.Adam(distil_policy.parameters())

# First parameter is the number of rollouts
imagination = ImaginationCore(1, state_shape, num_actions, num_rewards, env_model, distil_policy, full_rollout=full_rollout)

actor_critic = I2A(state_shape, num_actions, num_rewards, 256, imagination, full_rollout=full_rollout)
#rmsprop hyperparams:
lr    = 7e-4
eps   = 1e-5
alpha = 0.99

# Optimize parameters of I2A
optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

if USE_CUDA:
    env_model     = env_model.cuda()
    distil_policy = distil_policy.cuda()
    actor_critic  = actor_critic.cuda()

gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
num_steps = 5
num_frames = int(10e5)

rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
rollout.cuda()

all_rewards = []
all_losses  = []

# Get the initial state
state = envs.reset()
# Convert to torch tensor
current_state = torch.FloatTensor(np.float32(state))

# Set to the replay buffer
rollout.states[0].copy_(current_state)

episode_rewards = torch.zeros(num_envs, 1)
final_rewards   = torch.zeros(num_envs, 1)

print('Starting to train')
print('Using cuda?', USE_CUDA)

print('Using: %i GPUs' % (torch.cuda.device_count()))

for i_update in range(num_frames):
    overall_start = time.time()

    start = time.time()
    for step in range(num_steps):
        if USE_CUDA:
            current_state = current_state.cuda()

        # Get I2A action for state
        action = actor_critic.act(Variable(current_state))

        next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())

        reward = torch.FloatTensor(reward).unsqueeze(1)
        episode_rewards += reward
        masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1)
        final_rewards *= masks
        final_rewards += (1-masks) * episode_rewards
        episode_rewards *= masks

        if USE_CUDA:
            masks = masks.cuda()

        current_state = torch.FloatTensor(np.float32(next_state))
        rollout.insert(step, current_state, action.data, reward, masks)

    end = time.time()
    plog('Roll out takes', end - start)

    _, next_value = actor_critic(Variable(rollout.states[-1], volatile=True))
    next_value = next_value.data

    # Apply the bellman equation to calculate returns
    returns = rollout.compute_returns(next_value, gamma)

    # just the standard way of evaluating actions given state and action
    # This is for I2A
    logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
        Variable(rollout.states[:-1]).view(-1, *state_shape),
        Variable(rollout.actions).view(-1, 1)
    )

    # This is for the normal A2C
    distil_logit, _, _, _ = distil_policy.evaluate_actions(
        Variable(rollout.states[:-1]).view(-1, *state_shape),
        Variable(rollout.actions).view(-1, 1)
    )

    distil_loss = 0.01 * (F.softmax(logit, dim=1).detach() *
            F.log_softmax(distil_logit, dim=1)).sum(1).mean()

    values = values.view(num_steps, num_envs, 1)
    action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
    advantages = Variable(returns) - values

    value_loss = advantages.pow(2).mean()
    action_loss = -(Variable(advantages.data) * action_log_probs).mean()

    ###############################################
    ###############################################

    start = time.time()
    # Apparently before applying a gradient you have to zero out the gradients
    optimizer.zero_grad()
    loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
    loss.backward()
    nn.utils.clip_grad_norm(actor_critic.parameters(), max_grad_norm)
    optimizer.step()

    distil_optimizer.zero_grad()
    distil_loss.backward()
    optimizer.step()
    end = time.time()
    plog('Backpropagating took', end - start)

    overall_end = time.time()
    print('Epoch took', overall_end - overall_start)

    all_rewards.append(final_rewards.mean())
    all_losses.append(loss.data[0])
    print('Epoch %i, Rewards %.2f, Loss %.2f' % (i_update,
        np.mean(all_rewards[-10:]), all_losses[-1]))

    rollout.after_update()

    if i_update != 0 and i_update % 10000 == 0:
        print('Saving model!')
        torch.save(actor_critic.state_dict(), "i2a_" + mode + '_' + str(i_update))


torch.save(actor_critic.state_dict(), "i2a_" + mode)
