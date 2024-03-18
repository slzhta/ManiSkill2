import gymnasium as gym
from tqdm.notebook import tqdm
import numpy as np
import mani_skill.envs
import matplotlib.pyplot as plt
import os

env_id = "PickCube-v1"
obs_mode = "rgbd"
control_mode = "pd_joint_delta_pos"
reward_mode = "dense"

env = gym.make(env_id,
               obs_mode=obs_mode,
               reward_mode=reward_mode,
               control_mode=control_mode,
               enable_shadow=False)
obs, _ = env.reset()
print("Action Space:", env.action_space)

img = env.unwrapped.render_cameras()
plt.figure(figsize=(10,6))
plt.title("Current State viewed through all RGB and Depth cameras")
os.mkdir("./logs/pic")
plt.save("./logs/pic/test.png")
env.close()