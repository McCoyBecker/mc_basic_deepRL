import argparse
import sys
import time
import gym
from gym import wrappers, logger
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import scipy.signal

# Utilities.
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# Generates MLPs.
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis = 0)[::-1]

# This is essentially a nice interface to collect traces of experience, compute the right things for the loss.
class VPGBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma = 0.99, lam = 0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, 
                ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

# Actors + critic.
class Actor(nn.Module):
    def _dist(self, obs):
        raise NotImplementedError

    def _logprob_dist(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._dist(obs)
        logp_a = None
        if act is not None:
            logp_a = self._logprob_dist(pi, act)
        return pi, logp_a

class MLPGaussian(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _dist(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _logprob_dist(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

class MLPCategorical(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _dist(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _logprob_dist(self, pi, act):
        return pi.log_prob(act)

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        # Critic provides the value function.
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation= nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = MLPGaussian(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
    
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._dist(obs)
            a = pi.sample()
            logp_a = self.pi._logprob_dist(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

def vpg(env_fn, 
        actor_critic=MLPActorCritic, 
        seed = 314159,
        steps_per_epoch = 4000, 
        epochs = 50, 
        gamma = 0.99, 
        pi_lr = 3e-4, 
        vf_lr=1e-3, 
        train_v_iters = 80, 
        lam = 0.97, 
        max_ep_len = 1000):

    # Reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment.
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Actor-critic.
    ac = actor_critic(env.observation_space, env.action_space)

    # Buffer.
    local_steps_per_epoch = int(steps_per_epoch)
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss.
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        return loss_pi

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Optimizers.
    pi_opt = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_opt = Adam(ac.v.parameters(), lr = vf_lr)

    def update():
        data = buf.get()

        # Get loss before grad.
        pi_l_old = compute_loss_pi(data)
        v_l_old = compute_loss_v(data)
        pi_opt.zero_grad()

        # Grad step.
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_opt.step()

        # Value func.
        for i in range(train_v_iters):
            vf_opt.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_opt.step()

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            buf.store(o, a, r, v, logp)
            o = next_o

        update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    vpg(lambda : gym.make(args.env),
        actor_critic = MLPActorCritic,
        gamma = args.gamma,
        seed = args.seed,
        steps_per_epoch = args.steps,
        epochs = args. epochs)
