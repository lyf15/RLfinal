import model 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torch.distributions as dist
import utils
import copy

class agent1:
    def __init__(self, in_dim, out_dim, device, act_low, act_high, type_=1, eps=None, beta=None, dtarg=None):
        self.device = device
        self.act_low = torch.as_tensor(act_low, device=device)
        self.act_high = torch.as_tensor(act_high, device=device)
        self.Policy = model.MLP1(in_dim, out_dim).to(device)
        self.Value = model.MLP2(in_dim, 1).to(device)
        self.opt1 = optim.Adam(self.Policy.parameters(), lr=0.0003)
        self.opt2 = optim.Adam(self.Value.parameters(), lr=0.0003)
        self.type = type_
        self.eps = eps
        self.beta = beta
        self.dtarg = dtarg 
    
    def action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist, mean, std = self.Policy(obs)
            act = dist.sample().squeeze(0)
            logp = dist.log_prob(act).sum(dim=-1)
            val = self.Value(obs).squeeze(0)
            act2 = torch.max(torch.min(act, self.act_high), self.act_low)
            return act.cpu().numpy(), float(logp.item()), float(val.item()), act2.cpu().numpy(), mean.cpu().numpy(), std.cpu().numpy()

    def value(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.Value(obs).item())

    def update(self, all_obs, all_act, all_logp_old, all_mean_old, all_std_old, all_adv, all_ret, epochs=10, minibatch=64):
        T = all_obs.shape[0]
        idx = np.arange(T)
        loss1, loss2 = None, None
        # loss3, clip = [], []
        dist_final = 0.0
        all_mean_old, all_std_old = all_mean_old.squeeze(1), all_std_old.squeeze(1)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for st in range(0, T, minibatch): 
                batch = idx[st: st+minibatch]
                obs = all_obs[batch]
                act = all_act[batch]
                logp_old = all_logp_old[batch]
                mean_old = all_mean_old[batch]
                std_old = all_std_old[batch]
                adv = all_adv[batch]
                ret = all_ret[batch]


                # self.type = type_
                # self.eps = eps
                # self.beta = beta
                # self.dtarg = dtarg 
                dist_new, mean, std = self.Policy(obs)
                # print(std_old.shape)
                if self.type == 'r':
                    pi_backup = copy.deepcopy(self.Policy.state_dict())
                    opt_backup = copy.deepcopy(self.opt1.state_dict())
                    logp = dist_new.log_prob(act).sum(dim=-1)
                    ratio = torch.exp(logp-logp_old)
                    clip_ratio = torch.clamp(ratio, 1.0-self.eps, 1.0+self.eps)
                    loss1 = -(torch.min(ratio * adv, clip_ratio * adv)).mean()
                if self.type == 'c':
                    logp = dist_new.log_prob(act).sum(dim=-1)
                    ratio = torch.exp(logp-logp_old)
                    clip_ratio = torch.clamp(ratio, 1.0-self.eps, 1.0+self.eps)
                    loss1 = -(torch.min(ratio * adv, clip_ratio * adv)).mean()
                if self.type == 'a' or self.type == 'f':
                    dist_old = dist.Normal(mean_old, std_old)
                    logp = dist_new.log_prob(act).sum(dim=-1)
                    ratio = torch.exp(logp-logp_old)
                    KL = dist.kl_divergence(dist_old, dist_new)
                    KL = KL.sum(-1)
                    loss1 = -(ratio * adv - self.beta * KL).mean()          
                if self.type == 'n':
                    logp = dist_new.log_prob(act).sum(dim=-1)
                    ratio = torch.exp(logp-logp_old)
                    loss1 = -(ratio * adv).mean()
                if self.type == 'j':
                    dist_old = dist.Normal(mean_old, std_old)
                    logp = dist_new.log_prob(act).sum(dim=-1)
                    ratio = torch.exp(logp-logp_old)
                    # JS = utils.get_js_divergence(dist_old, dist_new, act)
                    JS = utils.get_js_divergence(dist_old, dist_new, 10240)
                    # print(ratio, ' ', adv, ' ', JS)
                    loss1 = -(ratio * adv - self.beta * JS).mean() 


                self.opt1.zero_grad(set_to_none=True)
                loss1.backward()
                nn.utils.clip_grad_norm_(self.Policy.parameters(), 0.5)
                self.opt1.step()

                val = self.Value(obs)
                loss2 = 0.5 * ((val - ret) ** 2).mean()
                self.opt2.zero_grad(set_to_none=True)
                loss2.backward()
                nn.utils.clip_grad_norm_(self.Value.parameters(), 0.5)
                self.opt2.step()

                if self.type == 'r':
                    with torch.no_grad():
                        low_rb  = 1.0 - 2.0 * self.eps
                        high_rb = 1.0 + 2.0 * self.eps
                        dist_post, __, __ = self.Policy(obs)
                        logp_post = dist_post.log_prob(act).sum(dim=-1)
                        ratio_post = torch.exp(logp_post - logp_old)
                        violated = (ratio_post < low_rb) | (ratio_post > high_rb)
                        frac_violated = violated.float().mean()
                        if frac_violated > 0.4:   
                            self.Policy.load_state_dict(pi_backup)
                            self.opt1.load_state_dict(opt_backup)
                            break
        
        if self.type == 'a':
            with torch.no_grad():
                dist_old = dist.Normal(all_mean_old, all_std_old)
                dist_new, mean, std = self.Policy(all_obs)
                KL = dist.kl_divergence(dist_old, dist_new)
                dist_final = KL.sum(-1).mean().item()
                if dist_final < self.dtarg / 1.5: self.beta = self.beta / 2.0
                if dist_final > self.dtarg * 1.5: self.beta = self.beta * 2.0 
        return loss1
