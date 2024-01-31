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

from envs.customfetch.custom_fetch import WallsDemoStackEnv

class ClipObsWrapper:
    def __init__(self, env, obs_min, obs_max):
        self._env = env
        self.obs_min = obs_min
        self.obs_max = obs_max

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        new_obs = np.clip(obs['observation'], self.obs_min, self.obs_max)
        obs['observation'] = new_obs
        return obs, rew, done, info

def wrap_mega_env(e, info_to_obs_fn=None):
    e = common.GymWrapper(e, info_to_obs_fn=info_to_obs_fn)
    if hasattr(e.act_space['action'], 'n'):
      e = common.OneHotAction(e)
    else:
      e = common.NormalizeAction(e)
    return e

class WallsDemoStackEnv_Image(WallsDemoStackEnv):
    """
    Converts the ThreeStack into image observations format.
    """
    def __init__(self, max_step=100, n=2, mode="-1/0", hard=False, distance_threshold=0.03, eval=False, initial_qpos=None, show_gripper=True, easy_mode=False):
        super().__init__(max_step=max_step, n=n, mode=mode, hard=hard, distance_threshold=distance_threshold, eval=eval, initial_qpos=initial_qpos) 
        
        self.show_gripper = show_gripper
        self.easy_mode = easy_mode

        observation_space = gym.spaces.Box(0, 1, (64, 64, 3))
        goal_space = gym.spaces.Box(0, 1, (64, 64, 3))
        self.observation_space = gym.spaces.Dict({
            'observation': observation_space,
            'desired_goal': goal_space,
            'achieved_goal': goal_space
        })

        self.all_goals_img = []
        for i in range(len(self.all_goals)):
            self.goal_idx = i
            self.all_goals_img.append(self.get_goal_render())

            
    def get_image_obs(self):
        sim = self.sim
        image_obs = sim.render(height=64, width=64, camera_name="external_camera_0")[::-1]
        image_obs = image_obs / 255
        #image_obs = cv2.resize(image_obs, (64, 64))
        #image_obs = image_obs.astype('float16')
        if self.easy_mode:
            image_obs[:32, :, :] = 0
        return image_obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        #TODO: NOT USED ACTUALLY, isntantly filtered out
        #obs['vector_state'] = obs['observation']
        obs['observation'] = self.get_image_obs()
        obs['achieved_goal'] = self.get_image_obs()
        obs['desired_goal'] = self.goal_image

        return obs, reward, done, info
  
    def get_goal_render(self):
        env = self
        sim = self.sim

        out_of_way_state = np.array([4.40000000e+00,  4.04999349e-01,  4.79999636e-01,  2.79652104e-06, 1.56722299e-02,-3.41500342e+00, 9.11469058e-02,-1.27681180e+00,
        -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
            2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
            1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
        -2.28652597e-07, 2.56090909e-07,-1.20181003e-15, 1.32999955e+00,
            8.49999274e-01, 4.24784489e-01, 1.00000000e+00,-2.77140579e-07,
            1.72443027e-07,-1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
            4.24784489e-01, 1.00000000e+00,-2.31485576e-07, 2.31485577e-07,
        -6.68816586e-16,-4.48284993e-08,-8.37398903e-09, 7.56100615e-07,
            5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
        -7.15601860e-02,-9.44665089e-02, 1.49646097e-02,-1.10990294e-01,
        -3.30174644e-03, 1.19462201e-01, 4.05130821e-04,-3.95036450e-04,
        -1.53880539e-07,-1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
        -6.18188284e-06, 1.31307184e-17,-1.03617993e-07,-1.66528917e-07,
            1.06089030e-14, 6.69000941e-06,-4.16267252e-06, 3.63225324e-17,
        -1.39095626e-07,-1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
        -5.58792469e-06,-2.07082526e-17])
        if not self.show_gripper:
            sim.set_state_from_flattened(out_of_way_state)
            sim.forward()
        else:
            state = self.initial_state
            sim.set_state(self.initial_state)
            sim.forward()

            gripper_xpos = sim.data.get_site_xpos('robot0:grip')
            gripper_target = np.array([-0.05, -0.07, 0]) + gripper_xpos
            sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            for _ in range(10):
                sim.step()

        sites_offset = (sim.data.site_xpos - sim.model.site_pos)
        site_id = sim.model.site_name2id('gripper_site')
        
        obs = self.all_goals[self.goal_idx]
        #obs = env.obs_min + obs * (env.obs_max -  env.obs_min)
        grip_pos = obs[:3]
        gripper_state = obs[3:5]
        all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
        # set the end effector site instead of the actual end effector.
        sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
        # set the objects
        for i, pos in enumerate(all_obj_pos):
          sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1,0,0,0]])

        sim.forward()
        
        return self.get_image_obs()

    def reset(self):
        obs = super().reset()

        # Compute the goal image

        self.goal_image = self.all_goals_img[self.goal_idx]
        
        obs['observation'] = self.get_image_obs()
        obs['achieved_goal'] = self.get_image_obs()
        obs['desired_goal'] = self.goal_image
        #print(obs['observation'].shape)
        #print(self.goal_image.shape)

        return obs

    # Functions needed to compute success
    # Get metrics function needs to be reimplemented because its used in info to obs
    def compute_reward_image(self, achieved_goal, goal, info):
        
        #print(achieved_goal.shape, goal.shape)
        errors = np.square(achieved_goal - goal).mean()
        
        success = errors < 0.01
        reward = success - 1 # maps to -1 if fail, 0 if success.
        return reward

    def add_pertask_success_image(self, info, obs, goal_idx = None):
        goal_idxs = [goal_idx] if goal_idx is not None else range(len(self.all_goals))
        for goal_idx in goal_idxs:
            g = self.all_goals_img[goal_idx]
            # compute normal success - if we reach within 0.15
            reward = self.compute_reward_image(obs, g, info)
            # -1 if not close, 0 if close.
            # map to 0 if not close, 1 if close.
            info[f"metric_success/goal_{goal_idx}"] = reward + 1
        return info

    def get_metrics_dict_image(self):
        ##print("GETTING THREE STACKS METRICS")
        ##print("a", self.observation_space['achieved_goal'].shape)
        info = {}
        dummy_obs = np.ones(self.observation_space['achieved_goal'].shape)
        ##print(dummy_obs.shape)
        if self.eval:
            info = self.add_pertask_success_image(info, dummy_obs, goal_idx=self.goal_idx)
            # by default set it to false.
            info[f"metric_success/goal_{self.goal_idx}"] = 0.0
        else:
            info = self.add_pertask_success_image(info, dummy_obs, goal_idx=None)
            for k,v in info.items():
                if 'metric' in k:
                    info[k] = 0.0
            z_threshold = 0.5
            for idx in range(self.n):
                info[f"metric_obj{idx}_above_{z_threshold:.2f}"] = 0.0
        return info

class ThreeStackEnvPEG():

    def __init__(self, config):
       self.config = config
       assert config.use_image

    def make_env(self, config,  use_goal_idx=False, log_per_goal=False, eval=False):
        """
        Create environments from LEXA benchmark or MEGA benchmark.
        use_goal_idx, log_per_goal are LEXA benchmark specific args.
        eval flag used for creating MEGA eval envs
        """
        env = WallsDemoStackEnv_Image(max_step=config.time_limit, eval=eval, n=3, easy_mode=config.easy_mode)

        # Fix obs keys
        env = common.ConvertGoalEnvWrapper(env)
        # LEXA assumes information is in obs dict already, so move info dict into obs.      
        def info_to_obs(info, obs):
            if info is None:
                info = env.get_metrics_dict_image()
            obs = obs.copy()
            for k,v in info.items():
                if eval:
                    if "metric" in k:
                        obs[k] = v
                else:
                    if "above" in k:
                        obs[k] = v
            return obs
        #info_to_obs = None

        env = wrap_mega_env(env, info_to_obs)
        return env

    # None
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
        from gym.envs.robotics import rotations
        def episode_render_fn(env, ep):
            sim = env.sim
            all_img = []
            goals = []
            executions = []

            all_img = ep['observation']

            goal_img = ep['goal'][0]
            goals.append(goal_img[None]) 

            #inner_env.goal = ep['goal'][0]
            
            #goals.append(all_img[0][None]) # 1 x H x W x C
            ep_img = np.stack(all_img, 0)
            T = ep_img.shape[0]
            ep_img = np.pad(ep_img, ((0, (self.config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'edge')

            executions.append(ep_img[None]) # 1 x T x H x W x C
            return goals, executions
            
        if self.config.no_render:
            episode_render_fn = None

        env = driver._envs[0]
        eval_goal_idxs = range(len(env.get_goals()))
        num_eval_eps = 10 
        executions = []
        goals = []
        all_metric_success = []
        # key is metric name, value is list of size num_eval_eps
        ep_metrics = collections.defaultdict(list)
        for ep_idx in range(num_eval_eps):
            print(f"Running eval {ep_idx+1}/{num_eval_eps}")
            should_video = ep_idx == 0 and episode_render_fn is not None
            for idx in eval_goal_idxs:
                driver.reset()
                # Set what goal to set at reset
                env.set_goal_idx(idx)
                # Reset and run eval policy
                driver(eval_policy, episodes=1)
                """ rendering based on state."""
                ep = driver._eps[0] # get episode data of 1st env.
                ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                score = float(ep['reward'].astype(np.float64).sum())
                print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
                # render the goal img and rollout
                for k, v in ep.items():
                    if 'metric_success/goal_' in k:
                        ep_metrics[k].append(np.max(v))
                        all_metric_success.append(np.max(v))

                if not should_video:
                    continue
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
        logger.scalar('mean_eval_metric_success/goal_all', all_metric_success)
        for key, value in ep_metrics.items():
            logger.scalar(f'mean_eval_{key}', np.mean(value))
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
    import cv2
    
    class FakeCFG():
        def __init__(self):
            self.use_image = True
            self.time_limit = 100
            self.no_render = False
            self.easy_mode = False
            self.show_gripper = True
    
    config = FakeCFG()
    env_wrapper = ThreeStackEnvPEG(config)
    env = env_wrapper.make_env(config)

    img2 = env.get_goal_render()
    img1 = env.get_goal_render()
    print(img1.min(), img1.max())

    cv2.imwrite("inway.png", (img1 * 255).astype(int)) 
    cv2.imwrite("outway.png", (img2 * 255).astype(int)) 
