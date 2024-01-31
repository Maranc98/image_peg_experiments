import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import dreamerv2.api as dv2
import common
from common import Config
import envs
import numpy as np
import collections
import matplotlib.pyplot as plt
from dreamerv2.common.replay import convert
import pathlib
import sys
import ruamel.yaml as yaml

import gym
import cv2

from envs.sibrivalry.toy_maze import MultiGoalPointMaze2D

def wrap_mega_env(e, info_to_obs_fn=None):
    e = common.GymWrapper(e, info_to_obs_fn=info_to_obs_fn)
    if hasattr(e.act_space['action'], 'n'):
        e = common.OneHotAction(e)
    else:
        e = common.NormalizeAction(e)
    return e

class MultiGoalPointMaze2D_Image(MultiGoalPointMaze2D):
    """
    Converts the MultiGoalPointMaze2D into image observations format.
    """
    def __init__(self, test=False):
        super().__init__(test)
        
        observation_space = gym.spaces.Box(0, 1, (64, 64, 3))
        goal_space = gym.spaces.Box(0, 1, (64, 64, 3))
        self.observation_space = gym.spaces.Dict({
            'observation': observation_space,
            'desired_goal': goal_space,
            'achieved_goal': goal_space
        })

        self.stored_goal = None

    def get_image_obs(self):
        image_obs = self.render()
        image_obs = image_obs / 255
        image_obs = cv2.resize(image_obs, (64, 64))
        #image_obs = image_obs.astype('float16')
        return image_obs


    def step(self, action):
        obs, reward, done, info = super().step(action)

        #TODO: NOT USED ACTUALLY, isntantly filtered out
        #obs['vector_state'] = obs['observation']
        obs['observation'] = self.get_image_obs()
        obs['achieved_goal'] = self.get_image_obs()
        obs['desired_goal'] = self.goal_image

        return obs, reward, done, info
  
    def reset(self):
        obs = super().reset()
        
        #obs['vector_state'] = obs['observation']
        obs['observation'] = self.get_image_obs()
        obs['achieved_goal'] = self.get_image_obs()

        #print("B4", obs['desired_goal'])
        temp_xy = self.s_xy 
        self.s_xy = self.g_xy
        self.goal_image = self.get_image_obs()
        self.s_xy = temp_xy
        obs['desired_goal'] = self.goal_image
        #print("AFTER", obs['desired_goal'])

        return obs

class PointMaze2DEnvPEG():

    def __init__(self, config):
        print("Using the Image Point Maze Environment")
        self.config = config
        assert config.use_image

    def make_env(self, config,  use_goal_idx=False, log_per_goal=False, eval=False):
        """
        Create environments from LEXA benchmark or MEGA benchmark.
        use_goal_idx, log_per_goal are LEXA benchmark specific args.
        eval flag used for creating MEGA eval envs
        """
        
        env = MultiGoalPointMaze2D_Image(test=eval)

        env.max_steps = config.time_limit
        # PointMaze2D is a GoalEnv, so rename obs dict keys.
        env = common.ConvertGoalEnvWrapper(env)
        # LEXA assumes information is in obs dict already, so move info dict into obs.
        info_to_obs = None
        if eval:
            def info_to_obs(info, obs):
                if info is None:
                    info = env.get_metrics_dict()
                obs = obs.copy()
                for k,v in info.items():
                    if "metric" in k:
                        obs[k] = v
                return obs
        env = wrap_mega_env(env, info_to_obs)
        class GaussianActions:
            """Add gaussian noise to the actions.
            """
            def __init__(self, env, std):
                self._env = env
                self.std = std

            def __getattr__(self, name):
                return getattr(self._env, name)

            def step(self, action):
                new_action = action
                if self.std > 0:
                    noise = np.random.normal(scale=self.std, size=2)
                    if isinstance(action, dict):
                        new_action = {'action': action['action'] + noise}
                    else:
                        new_action = action + noise

                return self._env.step(new_action)
        env = GaussianActions(env, std=0)
        return env

    def report_render_fn(self, recon, openl, truth, env):
        """
        Makes videos from states (i think)
        """
        # truth gets a +0.5
        # model gets a +0.5
        
        recon_imgs = recon.numpy()
        openl_imgs = openl.numpy()
        truth_imgs = truth.numpy()

        model = tf.concat([recon_imgs[:, :5], openl_imgs], 1)
        
        error = (model - truth_imgs + 1) / 2
        video = tf.concat([truth_imgs, model, error], 2)
        
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
  
    def episode_render_fn(self, env, ep):
        all_img = ep['observation']
        all_img = np.stack(all_img, 0) # T x H x W x C
        return all_img

    def eval_fn(self, driver, eval_policy, logger):
        def episode_render_fn(env, ep):
            all_img = []
            goals = []
            executions = []
            # Get ep and goal images
            goal_img = ep['goal'][0]
            all_img = ep['observation']
            # Add goal image to goals
            goals.append(goal_img[None]) # 1 x H x W x C
            ep_img = np.stack(all_img, 0)
            # pad if episode length is shorter than time limit.
            T = ep_img.shape[0]
            ep_img = np.pad(ep_img, ((0, (self.config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
            executions.append(ep_img[None]) # 1 x T x H x W x C
            return goals, executions
            
        env = driver._envs[0]
        num_goals = len(env.get_goals())
        num_eval_eps = 1
        executions = []
        goals = []
        all_metric_success = []
        all_metric_success_cell = []
        for ep_idx in range(num_eval_eps):
            should_video = ep_idx == 0 and episode_render_fn is not None
            for idx in range(num_goals):
                # This sets the idx of the goal
                # During the driver rollout, reset is called
                # The implemented reset gets the g_xy and gets a render from it
                env.set_goal_idx(idx)
                driver(eval_policy, episodes=1)
                if not should_video:
                    continue
                """ rendering based on state."""
                ep = driver._eps[0] # get episode data of 1st env.
                ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                # aggregate goal metrics across goals together.
                for k, v in ep.items():
                    if 'metric' in k:
                        if 'cell' in k.split('/')[0]:
                            all_metric_success_cell.append(np.max(v))
                        else:
                            all_metric_success.append(np.max(v))
                # render the goal img and rollout
                _goals, _executions = episode_render_fn(env, ep)
                goals.extend(_goals)
                executions.extend(_executions)

            if should_video:
                executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
                goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
                goals = np.repeat(goals, executions.shape[1], 1)
                gc_video = np.concatenate([goals, executions], -3)
                logger.video(f'eval_gc_policy', gc_video)
        all_metric_success = np.mean(all_metric_success)
        logger.scalar('max_eval_metric_success/goal_all', all_metric_success)
        all_metric_success_cell = np.mean(all_metric_success_cell)
        logger.scalar('max_eval_metric_success_cell/goal_all', all_metric_success_cell)
        logger.write()

    @property
    def cem_vis_fn(self):
       return None

    #def make plot fn
    @property
    def plot_fn(self):
       return None

    @property
    def obs2goal_fn(self):
       return None
    
    @property
    def sample_env_goals(self):
       return None

if __name__ == "__main__":
    class FakeCFG():
        def __init__(self):
            self.use_image = True
            self.time_limit = 100
    
    config = FakeCFG()
    env_wrapper = PointMaze2DEnvPEG(config)
    env = env_wrapper.make_env(config)
    print(env.get_goals())
    print(len(env.get_goals()))