import numpy as np
import torch
import torch.distributions as dist

def GAE(all_rew, all_val, all_don, all_tru, last_val, gamma, lam):
    adv = 0.0
    T = len(all_rew)
    all_adv = np.zeros(T)
    all_ret = np.zeros(T)
    for t in reversed(range(T)):
        nonterminal = 0 if (all_don[t]) else 1
        next_val = last_val[t]# if all_don[t] or t==T-1 else all_val[t+1] 
        delta = all_rew[t] + gamma * next_val - all_val[t]
        adv = delta + gamma * lam * nonterminal * adv
        all_adv[t] = adv

    
    all_ret = all_adv + all_val
    all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
    
    return all_ret, all_adv

class meanstd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon
    def update(self, x):
        mean_ = np.mean(x, axis=0)
        var_ = np.var(x, axis=0)
        count_ = x.shape[0]
        self.update_data(mean_, var_, count_)
    def update_data(self, mean_, var_, count_):
        delta = mean_ - self.mean
        count = self.count + count_ 
        self.mean = self.mean + delta * count_ / count
        m_a = self.var * self.count
        m_b = var_ * count_ 
        M = m_a + m_b + np.square(delta) * self.count * count_ / count 
        self.var = M / count 
        self.count = count
    def normalize(self, x, u=True):
        if u: self.update(x.reshape(1, -1))
        obs = np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)
        return obs


def get_js_divergence(p_dist, q_dist, K, action_samples=None):
    # samples = p_dist.sample((K,))
    # log_p = p_dist.log_prob(samples).sum(dim=-1)
    # log_q = q_dist.log_prob(samples).sum(dim=-1)
    # log_m = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - torch.log(torch.tensor(2.0))
    # js_val = 0.5 * (log_p - log_m) + 0.5 * (log_q - log_m)
    
    mu_p, sigma_p = p_dist.mean, p_dist.stddev
    mu_q, sigma_q = q_dist.mean, q_dist.stddev
    mu_m = (mu_p + mu_q) / 2
    sigma_m_sq = (sigma_p**2 + sigma_q**2) / 2 + ((mu_p - mu_q)**2) / 4
    sigma_m = torch.sqrt(sigma_m_sq)
    m_dist = dist.Normal(mu_m, sigma_m)
    kl_p_m = dist.kl_divergence(p_dist, m_dist)
    kl_q_m = dist.kl_divergence(q_dist, m_dist)
    js_val = 0.5 * kl_p_m + 0.5 * kl_q_m
    js_val = js_val.sum(-1)
    return js_val