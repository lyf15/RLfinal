import torch
import gym
import model
import agent
import logger
import logging
import utils 
import random
import numpy as np

def space_dim(space):
    return int(np.prod(space.shape))

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_random_score(env):
    done = False
    score = 0.0
    env.reset()

    while done != True:
        act = env.action_space.sample()
        new_obs, reward, done, _ = env.step(act)
        score += reward

    return score

def train1(seed_val, env_id, device, type_, eps=None, beta=None, dtarg=None):
    logger_ = logging.getLogger()
    logger_.info(f"\n============================{env_id}============================\n")
    env = gym.make(env_id)
    set_global_seeds(seed_val)
    env.seed(seed_val)
    env.action_space.seed(seed_val)
    in_dim = space_dim(env.observation_space)
    out_dim = space_dim(env.action_space)
    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)
    agent_ = agent.agent1(in_dim, out_dim, device, act_low, act_high, type_, eps, beta, dtarg)

    print(get_random_score(env))

    timestep = 1000000
    T = 2048
    t = 0
    obs = env.reset()
    obs_rms = utils.meanstd(obs.shape)
    score = 0.0
    all_scores = []
    len = 0
    

    while t <timestep:
        all_obs, all_act, all_logp_old, all_mean_old, all_std_old = [], [], [], [], []
        all_rew, all_val, all_don, all_tru, last_val = [], [], [], [], []
        done, truncated = None, None
        for i in range(T):
            obs = obs_rms.normalize(obs)
            act_raw, logp, val, act, mean, std = agent_.action(obs)
            
            new_obs, reward, done, info = env.step(act)
            truncated = info.get('TimeLimit.truncated', False)
            score += reward
            if done and (truncated==False): last_val_ = 0.0
            else: 
                n_obs = obs_rms.normalize(new_obs, u=False)
                last_val_ = agent_.value(n_obs)

            all_obs.append(obs)
            all_act.append(act_raw)
            all_logp_old.append(logp)          
            all_mean_old.append(mean)
            all_std_old.append(std)  
            all_rew.append(reward)
            all_val.append(val)
            all_don.append(done)
            all_tru.append(truncated)
            last_val.append(last_val_)
            
            t+=1
            obs = new_obs 
            if done:
                logger_.info(f"\nepisode: {len}, score: {score}")
                len += 1
                all_scores.append(score)
                score = 0.0
                obs = env.reset()
            
            if t >= timestep: break
        
        all_rew, all_val, all_don, all_tru = np.array(all_rew), np.array(all_val), np.array(all_don), np.array(all_tru)
        all_obs, all_act = np.array(all_obs), np.array(all_act)
        all_ret, all_adv = utils.GAE(all_rew, all_val, all_don, all_tru, last_val, 0.99, 0.95)
        all_mean_old, all_std_old = np.array(all_mean_old), np.array(all_std_old)
        
        all_obs = torch.as_tensor(all_obs, dtype=torch.float32, device=device)
        all_act = torch.as_tensor(all_act, dtype=torch.float32, device=device)  
        all_adv = torch.as_tensor(all_adv, dtype=torch.float32, device=device)
        all_ret = torch.as_tensor(all_ret, dtype=torch.float32, device=device)
        all_logp_old = torch.as_tensor(all_logp_old, dtype=torch.float32, device=device)
        all_mean_old = torch.as_tensor(all_mean_old, dtype=torch.float32, device=device)
        all_std_old = torch.as_tensor(all_std_old, dtype=torch.float32, device=device)

        loss1 = agent_.update(all_obs, all_act, all_logp_old, all_mean_old, all_std_old, all_adv, all_ret, epochs=10, minibatch=64)


    random_score = get_random_score(env)
    env.close()
    return random_score, all_scores
