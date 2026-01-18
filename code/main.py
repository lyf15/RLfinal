import train
import numpy as np
import argparse
import json
import logger
import logging

def get_score(random_score, scores):
    scores = np.asarray(scores, dtype=np.float64)
    score_mean = float(scores[-100:].mean())     
    print("mean: ", score_mean)     
    return score_mean

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', required=True)
    parser.add_argument('--device', '-d', required=True)
    parser.add_argument('--seed', '-s', type=int, required=True)
    parser.add_argument('--type', '-t', required=True) # clip: c; adaptive kl: a; fixed kl:f; no c/k: n;   
    parser.add_argument('--eps', type=float) # for type: clip/c
    parser.add_argument('--beta', type=float, default=1.0) # for type: f kl
    parser.add_argument('--dtarg', type=float) # for type: a kl
    parser.add_argument('--exp', '-x', type=int, required=True) # 1 for experiment 1
    parser.add_argument('--name', '-n', required=True) 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    if args.exp == 1:

        if args.env == 'all':
            logger.set_logger(f'{args.name}_seed_{args.seed}')
            env = ['HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
            results = {}
            score_dict = {}
            for env_id in env:
                random_score, scores = train.train1(args.seed, env_id, args.device, args.type, args.eps, args.beta, args.dtarg)
                score = get_score(random_score, scores)
                score_dict[f'{env_id}(random)'] = random_score
                score_dict[env_id] = score
            results[f"{args.name} with seed {args.seed}"] = score_dict
            with open(f'./results/{args.name}_seed_{args.seed}', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        else:
            random_score, scores = train.train1(args.seed, args.env, args.device, args.type, args.eps, args.beta, args.dtarg)
            scores = np.asarray(scores, dtype=np.float64)
            np.save(f'{args.name}_seed{args.seed}.npy', scores)
            get_score(random_score, scores)