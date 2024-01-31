# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Zelda env imports
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

import dreamerv2.api as dv2
import cv2
import numpy as np

import common
from common import Config

import tensorflow as tf

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
        observation = cv2.resize(observation, self.shape)
        return observation

class Zelda_GoalEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_steps = 50
        self.num_steps = 0
        
        observation_space = gym.spaces.Box(0, 1, (64, 64, 3))
        goal_space = gym.spaces.Box(0, 1, (64, 64, 3))
        self.observation_space = gym.spaces.Dict({
            'observation': observation_space,
            'desired_goal': goal_space,
            'achieved_goal': goal_space
        })

        self.stored_goal = None

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

        return obs_dict, reward, done, info
  
    def reset(self):
        obs = super().reset()
        obs = self.preprocess_obs(obs)

        #obs['vector_state'] = obs['observation']
        self.goal_image = obs
        # TODO This just skips making a meaningful goal
        obs_dict = {
            'observation': obs,
            'achieved_goal': obs,
            'desired_goal': self.goal_image,
        }
        #print("AFTER", obs['desired_goal'])

        self.num_steps = 0 

        return obs_dict

class PEG_Zelda():
    
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
        print("Using the Image Point Maze Environment")
        
        env = gym_zelda_1.make('Zelda1-v0')
        env = JoypadSpace(env, MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = ResizeObservation(env, shape=64)
        env = Zelda_GoalEnv(env)

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
        print("IMPLEMENT EVALUATION")

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

    

    env = gym_zelda_1.make('Zelda1-v0')
    env = JoypadSpace(env, MOVEMENT)

    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        print(f"{state.shape},\n {reward},\n {done},\n {info}")
        if step % 100 == 0:
            cv2.imwrite(f"examples/images/zelda/s{step}.png", state) 
        #env.render()

    env.close()


    1/0
    class FakeCFG():

        def __init__(self):
            self.use_image = True
            self.time_limit = 100

    config = FakeCFG()
    envpeg = PEG_Zelda(config)
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