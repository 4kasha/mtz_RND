import numpy as np
from collections import defaultdict, deque

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical

from model import CnnActorCritic, RNDModel
from utils import RunningMeanStd, RewardForwardFilter, RunningMeanStd_openAI

def get_advantage_and_value_target_from(reward, done, value, gamma, gae_lambda, num_step, num_worker, use_gae):
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * gae_lambda * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


class RNDagent(object):
    def __init__(self,
                 input_size,
                 output_size,
                 seed,
                 num_env,
                 pre_obs_norm_step,
                 num_step,
                 gamma=0.99,
                 gamma_int=0.99,
                 lam=0.95,
                 int_coef=1.,
                 ext_coef=2.,
                 ent_coef=0.001,
                 cliprange=0.1,
                 max_grad_norm=0.0,                
                 lr=1e-4,
                 nepochs=4,
                 batch_size=128,
                 update_proportion=0.25,
                 use_gae=True
                 ):

        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.seed = np.random.seed(seed)

        self.pre_obs_norm_step = pre_obs_norm_step
        self.num_step = num_step
        self.gamma = gamma
        self.gamma_int = gamma_int
        self.lam = lam
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.ent_coef = ent_coef
        self.cliprange = cliprange
        self.max_grad_norm = max_grad_norm
        self.update_proportion = update_proportion

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CnnActorCritic(input_size, output_size, seed).to(self.device)
        self.rnd = RNDModel(input_size, output_size, seed).to(self.device)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()), lr=lr)

        self.rff_int = RewardForwardFilter(gamma)
        #self.rff_rms_int = RunningMeanStd()
        #self.obs_rms = RunningMeanStd(shape=(1,84,84))
        self.rff_rms_int = RunningMeanStd_openAI()
        self.obs_rms = RunningMeanStd_openAI(shape=(1,84,84))

        self.rooms = None
        self.n_rooms = []
        self.best_nrooms = -np.inf
        self.scores = []
        self.scores_window = deque(maxlen=100) 
        
        self.stats = defaultdict(float) # Count episodes and timesteps
        self.stats['epcount'] = 0
        self.stats['tcount'] = 0                              

    def collect_random_statistics(self, envs):
        """Initializes observation normalization with data from random agent."""
        all_ob = []
        all_ob.append(envs.reset())
        for _ in range(self.pre_obs_norm_step):
            actions = np.random.randint(0, self.output_size, size=(self.num_env,))
            ob, _, _, _ = envs.step(actions)
            all_ob.append(ob)
            
            if len(all_ob) % (128 * self.num_env) == 0:
                ob_ = np.asarray(all_ob).astype(np.float32).reshape((-1, *envs.observation_space.shape))
                self.obs_rms.update(ob_[:,-1:,:,:])
                all_ob.clear()

    def act(self, state, action=None, calc_ent=False):
        """Returns dict of trajectory info.
        Shape
        ======
            state (uint8) : (batch_size, framestack=4, 84, 84)
        
        Returns example
            {'a': tensor([10,  5,  1]),
             'ent': None,
             'log_pi_a': tensor([-2.8904, -2.8904, -2.8904], grad_fn=<SqueezeBackward1>),
             'v_ext': tensor([0.0012, 0.0012, 0.0012], grad_fn=<SqueezeBackward0>),
             'v_int': tensor([-0.0013, -0.0013, -0.0013], grad_fn=<SqueezeBackward0>)}
        """
        #state = torch.FloatTensor(state / 255).to(self.device)
        assert state.dtype == 'uint8'
        state = torch.tensor(state / 255., dtype=torch.float, device=self.device)
        #state = torch.from_numpy(state /255).float().to(self.device)

        action_probs, value_ext, value_int = self.model(state)
        dist = Categorical(action_probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy() if calc_ent else None

        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v_ext': value_ext.squeeze(),
                'v_int': value_int.squeeze()}

    def compute_intrinsic_reward(self, next_obs):
        """next_obs is the latest frame and must be normalized by RunningMeanStd(shape=(1, 84, 84))
        Shape
        ======
            next_obs : (batch_size, 1, 84, 84)
        """
        next_obs = torch.tensor(next_obs, dtype=torch.float, device=self.device)
        #next_obs = torch.FloatTensor(next_obs).to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).mean(1)   ### MSE  --- Issues
        #intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def step(self, envs):
        """
        """
        # Step 1. n-step rollout
        next_obs_batch, int_reward_batch, state_batch, reward_batch, done_batch, action_batch, values_ext_batch, values_int_batch, log_prob_old_batch = [],[],[],[],[],[],[],[],[]
        epinfos = []
        
        states = envs.reset()
        for _ in range(self.num_step):

            traj_info = self.act(states)

            log_prob_old = traj_info['log_pi_a'].detach().cpu().numpy()
            actions = traj_info['a'].cpu().numpy()
            value_ext = traj_info['v_ext'].detach().cpu().numpy()
            value_int = traj_info['v_int'].detach().cpu().numpy()

            next_states, rewards, dones, infos = envs.step(actions)
            
            next_obs = next_states[:,-1:,:,:]
            intrinsic_reward = self.compute_intrinsic_reward(((next_obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var))).clip(-5, 5)) #+1e-10

            next_obs_batch.append(next_obs)
            int_reward_batch.append(intrinsic_reward)

            state_batch.append(states)
            reward_batch.append(rewards)
            done_batch.append(dones)
            action_batch.append(actions)
            values_ext_batch.append(value_ext)
            values_int_batch.append(value_int)
            log_prob_old_batch.append(log_prob_old)
            
            for info in infos:
                if 'episode' in info:
                    epinfos.append(info['episode'])

            states = next_states

        # calculate last next value
        last_traj_info = self.act(states)
        values_ext_batch.append(last_traj_info['v_ext'].detach().cpu().numpy())
        values_int_batch.append(last_traj_info['v_int'].detach().cpu().numpy())

        # convert to numpy array and transpose (num_env, num_step) from (num_step, num_env) for the later calculation
        # For self.update()
        state_batch = np.stack(state_batch).transpose(1, 0, 2, 3, 4).reshape(-1, 4, 84, 84)
        next_obs_batch = np.stack(next_obs_batch).transpose(1, 0, 2, 3, 4).reshape(-1, 1, 84, 84)
        action_batch = np.stack(action_batch).transpose().reshape(-1,)
        log_prob_old_batch = np.stack(log_prob_old_batch).transpose().reshape(-1,)
        
        # For get_advantage_and_value_target_from()
        reward_batch = np.stack(reward_batch).transpose()
        done_batch = np.stack(done_batch).transpose()
        values_ext_batch = np.stack(values_ext_batch).transpose()
        values_int_batch = np.stack(values_int_batch).transpose()
        # --------------------------------------------------

        # Step 2. calculate intrinsic reward
        # running estimate of the intrinsic returns
        int_reward_batch = np.stack(int_reward_batch).transpose()
        discounted_reward_per_env = np.array([self.rff_int.update(reward_per_step) for reward_per_step in int_reward_batch.T[::-1]])
        mean, std, count = np.mean(discounted_reward_per_env), np.std(discounted_reward_per_env), len(discounted_reward_per_env)
        self.rff_rms_int.update_from_moments(mean, std ** 2, count)   ### THINK ddof !

        # normalize intrinsic reward
        int_reward_batch /= np.sqrt(self.rff_rms_int.var)
        # -------------------------------------------------------------------------------------------

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = get_advantage_and_value_target_from(reward_batch,
                                                                  done_batch,
                                                                  values_ext_batch,
                                                                  self.gamma,
                                                                  self.lam,
                                                                  self.num_step,
                                                                  self.num_env,
                                                                  self.use_gae)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = get_advantage_and_value_target_from(int_reward_batch,
                                                                  np.zeros_like(int_reward_batch),
                                                                  values_int_batch,
                                                                  self.gamma_int,
                                                                  self.lam,
                                                                  self.num_step,
                                                                  self.num_env,
                                                                  self.use_gae)

        # add ext adv and int adv
        total_advs = self.int_coef * int_adv + self.ext_coef * ext_adv
        # -----------------------------------------------

        # Step 4. update obs normalize param
        self.obs_rms.update(next_obs_batch)
        # -----------------------------------------------

        # Step 5. Train
        loss_infos = self.update(state_batch, 
                                 ext_target, 
                                 int_target, 
                                 action_batch,
                                 total_advs,
                                 ((next_obs_batch - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var))).clip(-5, 5), #+1e-10
                                 log_prob_old_batch)
        # -----------------------------------------------

        # Collects info for reporting.
        vals_info = dict(
            advextmean = ext_adv.mean(),
            retextmean = ext_target.mean(),
            advintmean = int_adv.mean(),
            retintmean = int_target.mean(),
            rewintsample = int_reward_batch[1]  # env_number = 1
        )

        # Some reporting logic
        for epinfo in epinfos:
            #if self.testing:
            #    self.I.statlists['eprew_test'].append(epinfo['r'])
            #    self.I.statlists['eplen_test'].append(epinfo['l'])
            #else:
            if "visited_rooms" in epinfo:
                self.n_rooms.append(len(epinfo["visited_rooms"]))

                if self.best_nrooms is None:
                    self.best_nrooms = len(epinfo["visited_rooms"])
                elif len(epinfo["visited_rooms"]) > self.best_nrooms:
                    self.best_nrooms = len(epinfo["visited_rooms"])
                    self.rooms = sorted(list(epinfo["visited_rooms"]))
                #self.rooms += list(epinfo["visited_rooms"])
                #self.rooms = sorted(list(set(self.rooms)))
                #self.I.statlists['eprooms'].append(len(epinfo["visited_rooms"]))
            self.scores.append(epinfo['r'])
            self.scores_window.append(epinfo['r'])
            self.stats['epcount'] += 1
            self.stats['tcount'] += epinfo['l']
            #self.I.statlists['eprew'].append(epinfo['r'])
            #self.I.statlists['eplen'].append(epinfo['l'])
            #self.stats['rewtotal'] += epinfo['r']
        
        return {'loss' : loss_infos,
                'vals' : vals_info}

    def update(self, s_batch, target_ext_batch, target_int_batch, action_batch, adv_batch, next_obs_batch, log_prob_old_batch):
        #s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        log_prob_old_batch = torch.FloatTensor(log_prob_old_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        loss_infos = defaultdict(list)

        for _ in range(self.nepochs):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])
                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                
                # Proportion of exp used for predictor update   ---  cf. cnn_policy_param_matched.py
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).float().to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------

                traj_info = self.act(s_batch[sample_idx], action_batch[sample_idx], calc_ent=True)

                ratio = torch.exp(traj_info['log_pi_a'] - log_prob_old_batch[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange) * adv_batch[sample_idx]

                policy_loss = -torch.min(surr1, surr2).mean()

                critic_ext_loss = F.mse_loss(traj_info['v_ext'], target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(traj_info['v_int'], target_int_batch[sample_idx])
                value_loss = critic_ext_loss + critic_int_loss

                entropy = traj_info['ent'].mean()

                self.optimizer.zero_grad()
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy + forward_loss
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()), self.max_grad_norm)
                self.optimizer.step()
            
            _data = dict(
                policy = policy_loss.data.cpu().numpy(),
                value_ext = critic_ext_loss.data.cpu().numpy(),
                value_int = critic_int_loss.data.cpu().numpy(),
                entropy = entropy.data.cpu().numpy(),
                forward = forward_loss.data.cpu().numpy()
            )
            for k, v in _data.items():
                loss_infos[k].append(v)

        return loss_infos