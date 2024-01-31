# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

import dreamerv2.api as dv2
import cv2
import numpy as np

import common
from common import Config

import tensorflow as tf

import glob
import collections

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        #transforms = T.Compose(
        #    [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        #)
        #observation = transforms(observation).squeeze(0)
        observation = cv2.resize(observation, self.shape)
        return observation



class SuperMarioBros_GoalEnv(gym.Wrapper):
    def __init__(self, env, config, eval=False):
        super().__init__(env)
        self.config = config
        self.eval = eval

        self.max_steps = 50
        self.num_steps = 0
        
        observation_space = gym.spaces.Box(0, 1, (64, 64, 3))
        goal_space = gym.spaces.Box(0, 1, (64, 64, 3))
        self.observation_space = gym.spaces.Dict({
            'observation': observation_space,
            'desired_goal': goal_space,
            'achieved_goal': goal_space
        })
        
        # Loads goal images
        goal_filenames = glob.glob("./*/mario_goals/goal_*.png")
        goal_filenames.sort(key=lambda f: int(f.split("mario_goals/goal_")[-1][:-4]))
        self.all_goals = [cv2.imread(filename) for filename in goal_filenames]
        self.all_goals = [cv2.resize(img, (64,64)) for img in self.all_goals]
        self.all_goals = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.all_goals]
        self.all_goals = [img.astype(float) / 255 for img in self.all_goals]
        
        self.goal_idx = 0
        self.goal_image = None

    def preprocess_obs(self, obs):
        obs = obs / 255
        #obs = cv2.resize(obs, (64, 64))
        #obs = obs.astype('float16')
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        obs = self.preprocess_obs(obs)
        obs_dict = {
            'observation': obs,
            'achieved_goal': obs,
            'desired_goal': self.goal_image,
        }
        
        self.num_steps += 1
        if self.num_steps >= self.max_steps and not done:
            done = True
            info['TimeLimit.truncated'] = True

        # Compute if the goal is ahcieved
        # All timesteps need to have the same keys, so you need to fill in for unused goals too
        img_error = np.square(obs_dict['achieved_goal'] - obs_dict['desired_goal']).mean()
        success = img_error < 0.01
        if self.eval:
            info[f"metric_success/goal_{self.goal_idx}"] = 1.0 if success else 0.0
        else:
            for i in range(len(self.all_goals)):
                info[f"metric_success/goal_{i}"] = 0.0 
            info[f"metric_success/goal_{self.goal_idx}"] = 1.0 if success else 0.0

        return obs_dict, reward, done, info
  
    def reset(self):
        obs = super().reset()
        obs = self.preprocess_obs(obs)
        
        self.goal_image = self.all_goals[self.goal_idx]
        
        obs_dict = {
            'observation': obs,
            'achieved_goal': obs,
            'desired_goal': self.goal_image,
        }
        #print("AFTER", obs['desired_goal'])

        self.num_steps = 0 

        return obs_dict

    # Interfaces for evaluation functions
    def get_goals(self):
        return self.all_goals
    
    def set_goal_idx(self, idx):
        self.goal_idx = idx

    def get_metrics_dict(self):
        info = {}
        if self.eval:
            info[f"metric_success/goal_{self.goal_idx}"] = 0.0 
        else:
            for i in range(len(self.all_goals)):
                info[f"metric_success/goal_{i}"] = 0.0 

        return info

class PEG_SuperMarioBros():
    
    def __init__(self, config):
       self.config = config
       #assert config.use_image

    def make_env(self, config,  use_goal_idx=False, log_per_goal=False, eval=False):
        """
        Creates the SMB env
        use_goal_idx, log_per_goal are LEXA benchmark specific args.
        eval flag used for creating MEGA eval envs
        """
        def wrap_mega_env(e, info_to_obs_fn=None):
            e = common.GymWrapper(e, info_to_obs_fn=info_to_obs_fn)
            if hasattr(e.act_space['action'], 'n'):
                e = common.OneHotAction(e)
            else:
                e = common.NormalizeAction(e)
            return e
        
        assert config.use_image
        print("Using the Image Super Mario Environment")
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = ResizeObservation(env, shape=64)
        env = SuperMarioBros_GoalEnv(env, config, eval=eval)

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
            
            all_img = ep['observation']
            goal_img = ep['goal'][0]
            goals.append(goal_img[None]) 

            ep_img = np.stack(all_img, 0)
            T = ep_img.shape[0]
            ep_img = np.pad(ep_img, ((0, (self.config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'edge')

            executions.append(ep_img[None]) # 1 x T x H x W x C
            return goals, executions

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
    class FakeCFG():

        def __init__(self):
            self.use_image = True
            self.time_limit = 100

    config = FakeCFG()
    envpeg = PEG_SuperMarioBros(config)
    env = envpeg.make_env(config)
    print(env.action_space,env.observation_space)

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    envw = SuperMarioBros_GoalEnv(env)
    print(envw.action_space)
    print([envw.action_space.sample() for i in range(10)])
    print(envw.action_space.n)
    
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)

    if False:
        done = True
        for step in range(5000):
            if done:
                state = env.reset()
            state, reward, done, info = env.step(env.action_space.sample())
            print(f"{state.shape},\n {reward},\n {done},\n {info}")
            if step % 100 == 0:
                cv2.imwrite(f"examples/images/s{step}.png", state) 
            #env.render()

        env.close()