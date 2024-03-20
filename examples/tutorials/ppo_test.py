import gymnasium as gym
import gymnasium.spaces as spaces
from tqdm.notebook import tqdm
import numpy as np
import mani_skill.envs
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as th
import multiprocessing
import os
from matplotlib.animation import FuncAnimation

class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info

# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, is_vecenv_wrapped
from mani_skill.utils.wrappers import RecordEpisode
from stable_baselines3.common.utils import set_random_seed

# define an SB3 style make_env function for evaluation
def make_env(env_id: str, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill.envs
        env = gym.make(env_id, 
                       obs_mode=obs_mode, 
                       reward_mode=reward_mode, 
                       control_mode=control_mode, 
                       max_episode_steps=max_episode_steps, 
                       robot_uids="xarm7_five", # Try to build this robot
                       render_mode="open3d")
        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env)
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisode(
                env, record_dir, info_on_video=True
            )
        return env
    return _init

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from stable_baselines3 import PPO

from stable_baselines3.common.evaluation import evaluate_policy

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from stable_baselines3.common import base_class

finger_name = ["thumb", "index", "middle", "ring", "pinky"]

def plot_the_hand(save_dir, info):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(info["thumb_base"][0], info["thumb_base"][1], info["thumb_base"][2], c='r', marker='o')

    for finger in finger_name:
        ax.scatter(info[finger + "_L1"][0], info[finger + "_L1"][1], info[finger + "_L1"][2], c='r', marker='o')
        ax.scatter(info[finger + "_L2"][0], info[finger + "_L2"][1], info[finger + "_L2"][2], c='r', marker='o')
        ax.scatter(info[finger + "_tip"][0], info[finger + "_tip"][1], info[finger + "_tip"][2], c='r', marker='o')

        ax.plot((info["thumb_base"][0], info[finger + "_L1"][0]), 
                (info["thumb_base"][1], info[finger + "_L1"][1]), 
                (info["thumb_base"][2], info[finger + "_L1"][2]), 
                c='r', label='Lines')
        ax.plot((info[finger + "_L1"][0], info[finger + "_L2"][0]), 
                (info[finger + "_L1"][1], info[finger + "_L2"][1]), 
                (info[finger + "_L1"][2], info[finger + "_L2"][2]), 
                c='r', label='Lines')
        ax.plot((info[finger + "_L2"][0], info[finger + "_tip"][0]), 
                (info[finger + "_L2"][1], info[finger + "_tip"][1]), 
                (info[finger + "_L2"][2], info[finger + "_tip"][2]), 
                c='r', label='Lines')

    plt.savefig(save_dir)
    plt.close()

def plot_with_video(save_dir, infos, r_infos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        info = infos[frame]
        reward = r_infos[frame]
        ax.scatter(info["thumb_base"][0], info["thumb_base"][1], info["thumb_base"][2], c='b', marker='o')

        for finger in finger_name:
            ax.scatter(info[finger + "_L1"][0], info[finger + "_L1"][1], info[finger + "_L1"][2], c='b', marker='o')
            ax.scatter(info[finger + "_L2"][0], info[finger + "_L2"][1], info[finger + "_L2"][2], c='b', marker='o')
            ax.scatter(info[finger + "_tip"][0], info[finger + "_tip"][1], info[finger + "_tip"][2], c='b', marker='o')

            ax.plot((info["thumb_base"][0], info[finger + "_L1"][0]), 
                    (info["thumb_base"][1], info[finger + "_L1"][1]), 
                    (info["thumb_base"][2], info[finger + "_L1"][2]), 
                    c='r', label='Lines')
            ax.plot((info[finger + "_L1"][0], info[finger + "_L2"][0]), 
                    (info[finger + "_L1"][1], info[finger + "_L2"][1]), 
                    (info[finger + "_L1"][2], info[finger + "_L2"][2]), 
                    c='r', label='Lines')
            ax.plot((info[finger + "_L2"][0], info[finger + "_tip"][0]), 
                    (info[finger + "_L2"][1], info[finger + "_tip"][1]), 
                    (info[finger + "_L2"][2], info[finger + "_tip"][2]), 
                    c='r', label='Lines')

        ax.scatter(info["link_base"][0], info["link_base"][1], info["link_base"][2], c='b', marker='o')
        ax.scatter(info["link1"][0], info["link1"][1], info["link1"][2], c='b', marker='o')
        ax.scatter(info["link2"][0], info["link2"][1], info["link2"][2], c='b', marker='o')
        ax.scatter(info["link3"][0], info["link3"][1], info["link3"][2], c='b', marker='o')
        ax.scatter(info["link4"][0], info["link4"][1], info["link4"][2], c='b', marker='o')
        ax.scatter(info["link5"][0], info["link5"][1], info["link5"][2], c='b', marker='o')
        ax.scatter(info["link6"][0], info["link6"][1], info["link6"][2], c='b', marker='o')
        ax.scatter(info["link7"][0], info["link7"][1], info["link7"][2], c='b', marker='o')
        ax.scatter(info["base"][0], info["base"][1], info["base"][2], c='y', marker='o')
        ax.scatter(info["goal_pos"][0], info["goal_pos"][1], info["goal_pos"][2], c='g', marker='o')
        ax.scatter(info["obj_pose"][0], info["obj_pose"][1], info["obj_pose"][2], c='r', marker='o')

        ax.plot((info["link_base"][0], info["link1"][0]), 
                (info["link_base"][1], info["link1"][1]), 
                (info["link_base"][2], info["link1"][2]), 
                c='r', label='Lines')
        ax.plot((info["link1"][0], info["link2"][0]), 
                (info["link1"][1], info["link2"][1]), 
                (info["link1"][2], info["link2"][2]), 
                c='r', label='Lines')
        ax.plot((info["link2"][0], info["link3"][0]), 
                (info["link2"][1], info["link3"][1]), 
                (info["link2"][2], info["link3"][2]), 
                c='r', label='Lines')
        ax.plot((info["link3"][0], info["link4"][0]), 
                (info["link3"][1], info["link4"][1]), 
                (info["link3"][2], info["link4"][2]), 
                c='r', label='Lines')
        ax.plot((info["link4"][0], info["link5"][0]), 
                (info["link4"][1], info["link5"][1]), 
                (info["link4"][2], info["link5"][2]), 
                c='r', label='Lines')
        ax.plot((info["link5"][0], info["link6"][0]), 
                (info["link5"][1], info["link6"][1]), 
                (info["link5"][2], info["link6"][2]), 
                c='r', label='Lines')
        ax.plot((info["link6"][0], info["link7"][0]), 
                (info["link6"][1], info["link7"][1]), 
                (info["link6"][2], info["link7"][2]), 
                c='r', label='Lines')
        ax.plot((info["link7"][0], info["thumb_base"][0]), 
                (info["link7"][1], info["thumb_base"][1]), 
                (info["link7"][2], info["thumb_base"][2]), 
                c='r', label='Lines')
        # ax.plot((info["base"][0], info["thumb_base"][0]), 
        #         (info["base"][1], info["thumb_base"][1]), 
        #         (info["base"][2], info["thumb_base"][2]), 
        #         c='r', label='Lines')

        text_annotation1 = 'reward: {:6f}'.format(reward[0])
        ax.text(0.05, -2, 6.8, text_annotation1, transform=ax.transAxes, fontsize=12, color='black')
        text_annotation2 = 'cube: ({:3f}, {:.3f}, {:.3f})'.format(info['obj_pose'][0], info['obj_pose'][1], info['obj_pose'][2])
        ax.text(0.05, -2, 6.5, text_annotation2, transform=ax.transAxes, fontsize=12, color='black')


        ax.set_title('Frame: {}'.format(frame))
        ax.set_xlim(-0.2, 0.6)
        ax.set_ylim(-0.4, 0.4)
        ax.set_zlim(0, 0.8)
    
    num_frames = len(infos)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=num_frames)
    animation.save(save_dir, writer='ffmpeg', fps=20)

def plot_with_video_five(save_dir, infos, r_infos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        info = infos[frame]
        reward = r_infos[frame]
        ax.scatter(info["Link111"][0], info["Link111"][1], info["Link111"][2], c='b', marker='o')

        for finger in ["Link1", "Link2", "Link3", "Link4"]:
            ax.scatter(info[finger][0], info[finger][1], info[finger][2], c='b', marker='o')
            ax.scatter(info[finger + finger[-1]][0], info[finger + finger[-1]][1], info[finger + finger[-1]][2], c='b', marker='o')
            # ax.scatter(info[finger + "_tip"][0], info[finger + "_tip"][1], info[finger + "_tip"][2], c='b', marker='o')

            ax.plot((info["Link111"][0], info[finger][0]), 
                    (info["Link111"][1], info[finger][1]), 
                    (info["Link111"][2], info[finger][2]), 
                    c='r', label='Lines')
            ax.plot((info[finger][0], info[finger + finger[-1]][0]), 
                    (info[finger][1], info[finger + finger[-1]][1]), 
                    (info[finger][2], info[finger + finger[-1]][2]), 
                    c='r', label='Lines')
            # ax.plot((info[finger + "_L2"][0], info[finger + "_tip"][0]), 
            #         (info[finger + "_L2"][1], info[finger + "_tip"][1]), 
            #         (info[finger + "_L2"][2], info[finger + "_tip"][2]), 
            #         c='r', label='Lines')

        ax.scatter(info["Link5"][0], info["Link5"][1], info["Link5"][2], c='b', marker='o')
        ax.scatter(info["Link51"][0], info["Link51"][1], info["Link51"][2], c='b', marker='o')
        ax.scatter(info["Link52"][0], info["Link52"][1], info["Link52"][2], c='b', marker='o')
        ax.scatter(info["Link53"][0], info["Link53"][1], info["Link53"][2], c='b', marker='o')

        ax.plot((info["Link111"][0], info["Link5"][0]), 
                (info["Link111"][1], info["Link5"][1]), 
                (info["Link111"][2], info["Link5"][2]), 
                c='r', label='Lines')
        ax.plot((info["Link5"][0], info["Link51"][0]), 
                (info["Link5"][1], info["Link51"][1]), 
                (info["Link5"][2], info["Link51"][2]), 
                c='r', label='Lines')
        ax.plot((info["Link51"][0], info["Link52"][0]), 
                (info["Link51"][1], info["Link52"][1]), 
                (info["Link51"][2], info["Link52"][2]), 
                c='r', label='Lines')
        ax.plot((info["Link52"][0], info["Link53"][0]), 
                (info["Link52"][1], info["Link53"][1]), 
                (info["Link52"][2], info["Link53"][2]), 
                c='r', label='Lines')

        # ax.scatter(info["link_base"][0], info["link_base"][1], info["link_base"][2], c='b', marker='o')
        # ax.scatter(info["link1"][0], info["link1"][1], info["link1"][2], c='b', marker='o')
        # ax.scatter(info["link2"][0], info["link2"][1], info["link2"][2], c='b', marker='o')
        # ax.scatter(info["link3"][0], info["link3"][1], info["link3"][2], c='b', marker='o')
        # ax.scatter(info["link4"][0], info["link4"][1], info["link4"][2], c='b', marker='o')
        # ax.scatter(info["link5"][0], info["link5"][1], info["link5"][2], c='b', marker='o')
        # ax.scatter(info["link6"][0], info["link6"][1], info["link6"][2], c='b', marker='o')
        # ax.scatter(info["link7"][0], info["link7"][1], info["link7"][2], c='b', marker='o')
        ax.scatter(info["base"][0], info["base"][1], info["base"][2], c='y', marker='o')
        # ax.scatter(info["goal_pos"][0], info["goal_pos"][1], info["goal_pos"][2], c='g', marker='o')
        # ax.scatter(info["obj_pose"][0], info["obj_pose"][1], info["obj_pose"][2], c='r', marker='o')

        # ax.plot((info["link_base"][0], info["link1"][0]), 
        #         (info["link_base"][1], info["link1"][1]), 
        #         (info["link_base"][2], info["link1"][2]), 
        #         c='r', label='Lines')
        # ax.plot((info["link1"][0], info["link2"][0]), 
        #         (info["link1"][1], info["link2"][1]), 
        #         (info["link1"][2], info["link2"][2]), 
        #         c='r', label='Lines')
        # ax.plot((info["link2"][0], info["link3"][0]), 
        #         (info["link2"][1], info["link3"][1]), 
        #         (info["link2"][2], info["link3"][2]), 
        #         c='r', label='Lines')
        # ax.plot((info["link3"][0], info["link4"][0]), 
        #         (info["link3"][1], info["link4"][1]), 
        #         (info["link3"][2], info["link4"][2]), 
        #         c='r', label='Lines')
        # ax.plot((info["link4"][0], info["link5"][0]), 
        #         (info["link4"][1], info["link5"][1]), 
        #         (info["link4"][2], info["link5"][2]), 
        #         c='r', label='Lines')
        # ax.plot((info["link5"][0], info["link6"][0]), 
        #         (info["link5"][1], info["link6"][1]), 
        #         (info["link5"][2], info["link6"][2]), 
        #         c='r', label='Lines')
        # ax.plot((info["link6"][0], info["link7"][0]), 
        #         (info["link6"][1], info["link7"][1]), 
        #         (info["link6"][2], info["link7"][2]), 
        #         c='r', label='Lines')
        ax.plot((info["link7"][0], info["Link111"][0]), 
                (info["link7"][1], info["Link111"][1]), 
                (info["link7"][2], info["Link111"][2]), 
                c='r', label='Lines')

        # text_annotation1 = 'reward: {:6f}'.format(reward[0])
        # ax.text(0.05, -2, 6.8, text_annotation1, transform=ax.transAxes, fontsize=12, color='black')
        # text_annotation2 = 'cube: ({:3f}, {:.3f}, {:.3f})'.format(info['obj_pose'][0], info['obj_pose'][1], info['obj_pose'][2])
        # ax.text(0.05, -2, 6.5, text_annotation2, transform=ax.transAxes, fontsize=12, color='black')


        ax.set_title('Frame: {}'.format(frame))
        # ax.set_xlim(-0.2, 0.6)
        # ax.set_ylim(-0.4, 0.4)
        # ax.set_zlim(0, 0.8)
    
    num_frames = len(infos)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=num_frames)
    animation.save(save_dir, writer='ffmpeg', fps=20)

def make_evaluation(model, env, id, env_id):
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    n_envs = env.num_envs
    n_eval_episodes = 1
    deterministic = True
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    # print("Initial observation:", observations)

    if not os.path.exists("./logs/render/pic"):
        os.makedirs("./logs/render/pic")

    hand_informations = []
    reward_informations = []

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)

        reward_informations.append(rewards)

        # print("---------------------Step:{}---------------------".format(current_lengths + 1))
        # print("Action:", actions)
        # print("Reward:", rewards)
        # print("Dones:", dones)
        # print("Infomations:", infos)
        # print("Next obertvation:", observations)

        # plot_the_hand("./logs/render/pic/{}.png".format(current_lengths[0]), infos[0])
        hand_informations.append(infos[0])

        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    plot_with_video_five("./logs/render/five-{}-{}.mp4".format(env_id, id), hand_informations, reward_informations)

    print("Mean_reward:", mean_reward, "Reward_std:", std_reward)


if __name__=="__main__":
    
    multiprocessing.set_start_method('spawn')

    num_envs = 16 # you can increases this and decrease the n_steps parameter if you have more cores to speed up training
    env_id = "PushCube-v1"
    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "normalized_dense" # this the default reward mode which is a dense reward scaled to [0, 1]

    # create one eval environment
    eval_env = SubprocVecEnv([make_env(env_id, record_dir="./logs/vedio") for i in range(1)])
    eval_env = VecMonitor(eval_env) # attach this so SB3 can log reward metrics
    eval_env.seed(0)
    eval_env.reset()

    # create num_envs training environments
    # we also specify max_episode_steps=50 to speed up training
    env = SubprocVecEnv([make_env(env_id, max_episode_steps=50) for i in range(num_envs)])
    env = VecMonitor(env)
    env.seed(0)
    obs = env.reset()

    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=32000,
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=32000,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    set_random_seed(0) # set SB3's global seed to 0
    rollout_steps = 3200

    # create our model
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
        n_steps=rollout_steps // num_envs, batch_size=1024,
        n_epochs=16,
        tensorboard_log="./logs",
        gamma=0.9,
        target_kl=0.05
    )

    print("Finish prepare.")

    # Train with PPO
    model.learn(10_000, callback=[checkpoint_callback, eval_callback])
    model.save("./logs/latest_model")

    # optionally load back the model that was saved
    # model = model.load("./logs/latest_model")

    print("Finish train.")

    eval_env.close() # close the old eval env
    # make a new one that saves to a different directory
    eval_env = SubprocVecEnv([make_env(env_id, record_dir="logs/eval_videos") for i in range(1)])
    # eval_env = SubprocVecEnv([make_env(env_id, record_dir="logs/eval_videos") for i in range(1)])
    eval_env = VecMonitor(eval_env) # attach this so SB3 can log reward metrics
    eval_env.seed(1)
    eval_env.reset()

    # Do self-writen evaluation to get more information
    # for i in range(20):
    #     make_evaluation(model, eval_env, i, env_id) 

    # Use sb3 eval to check if eval is correct
    returns, ep_lens = evaluate_policy(model, eval_env, deterministic=True, render=False, return_episode_rewards=True, n_eval_episodes=10)
    print(f"Returns: {returns}")
    print(f"Episode Lengths: {ep_lens}")
    # success = np.array(ep_lens) < 200 # episode length < 200 means we solved the task before time ran out
    # success_rate = success.mean()
    # print(f"Success Rate: {success_rate}")
    # print(f"Episode Lengths: {ep_lens}")

    # Current can not display video
    # from IPython.display import Video
    # Video("./logs/eval_videos/2.mp4", embed=True) # Watch one of the replays
