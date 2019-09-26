"""
Exploration by Random Network Distillation, Pytorch implementation.
cf. https://arxiv.org/abs/1810.12894

"""

import numpy as np
from collections import defaultdict, deque
import pickle
import os
import torch

from agent import RNDagent
from parallel_envs.make_atari import make_atari_env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():
    parser = arg_parser()
    parser.add_argument('--env_id', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--seed_gym', help='seed for gym', type=int, default=0)
    #parser.add_argument('--max_episode_steps', type=int, default=4500)

    parser.add_argument('--num_timesteps', type=int, default=int(1e8)) # 1e12
    parser.add_argument('--num_env', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)  # 0.999
    parser.add_argument('--gamma_int', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--proportion_of_exp_used_for_predictor_update', type=float, default=1.)
    parser.add_argument('--int_coef', type=float, default=1.)
    parser.add_argument('--ext_coef', type=float, default=2.)

    args = parser.parse_args()

    model_path = '{}/weights/{}.model'.format(os.getcwd(), args.env_id)
    predictor_path = '{}/weights/{}.pred'.format(os.getcwd(), args.env_id)
    target_path = '{}/weights/{}.target'.format(os.getcwd(), args.env_id)
    scores_file='{}/scores_rnd.txt'.format(os.getcwd())

    best_score = -np.inf
    
    ########  Environment Setting  ########
    envs = make_atari_env(env_id=args.env_id, num_env=args.num_env, seed=args.seed_gym)

    print('Number of envs:', args.num_env)
    state_size = envs.observation_space.shape
    print('state size:', state_size)
    action_size = envs.action_space.n
    print('action size:', action_size)
    #######################################

    ###########  Agent Setting  ###########
    agent = RNDagent(input_size=envs.observation_space.shape,
                     output_size=envs.action_space.n,
                     seed=0,
                     num_env=envs.num_envs,
                     pre_obs_norm_step=128*50,
                     num_step=128,
                     gamma=args.gamma,
                     gamma_int=args.gamma_int,
                     lam=args.lam,
                     int_coef=args.int_coef,
                     ext_coef=args.ext_coef,
                     ent_coef=0.001,
                     cliprange=0.1,
                     max_grad_norm=0.0,                
                     lr=1e-4,
                     nepochs=4,
                     batch_size=128,
                     update_proportion=0.1, #0.25
                     use_gae=True
                    )

    print('-------- Model structure --------')
    print('-------- Model --------')
    print(agent.model)
    print('-------- RND -------')
    print(agent.rnd)
    print('---------------------------------')  
    #######################################

    global_update = 0
    global_step = 0
    loss = defaultdict(list)

    # normalize obs
    print('Start to initialize observation normalization parameter.....')
    agent.collect_random_statistics(envs)
    print('Done.')

    while True:
        info = agent.step(envs)

        #for k, v in info.items():
        #    loss[k].extend(v)
        global_update += 1
        global_step += (args.num_env * 128)
        print('\rupdate steps: {}\tEpisode {}\tAverage Score: {:.3f}\tVisited Room: {}'.format(global_update, agent.stats['epcount'], np.mean(agent.scores_window), agent.rooms), end="")
        
        if agent.stats['epcount'] % 100 == 0:
            print('\rglobal steps: {}\tEpisode {}\tAverage Score: {:.3f}\tVisited Room: {}'.format(global_step, agent.stats['epcount'], np.mean(agent.scores_window), agent.rooms))

        if np.mean(agent.scores_window) > best_score:
            print('\rsave.')
            print('\rupdate steps: {}\tEpisode {}\tAverage Score: {:.3f}\tVisited Room: {}'.format(global_update, agent.stats['epcount'], np.mean(agent.scores_window), agent.rooms))
            
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)
            best_score = np.mean(agent.scores_window)

        if agent.stats['tcount'] > args.num_timesteps:
            f = open(scores_file, 'wb')
            pickle.dump(agent.scores, f)
            break

    f.close()
    envs.close()

if __name__ == '__main__':
    main()