#!/usr/bin/env python3

import os
import numpy as np
import math
import gym
import gym_miniworld
from gym_miniworld.wrappers import PyTorchObsWrapper, GreyscaleWrapper
from gym_miniworld.entity import TextFrame
import matplotlib.pyplot as plt

env = gym.make('MiniWorld-Hallway-v0')

# Try stepping a few times
for i in range(0, 10):
    obs, r, done, info = env.step(0) # obs.shape = (60,80,3)
    # print(obs,r,done,info)

# Check that the human rendering resembles the agent's view
# 检查人类渲染视角和智能体视角相似程度
first_obs = env.reset() # first_obs.shape = (60,80,3)
first_render = env.render('rgb_array') # first_render.shape = (600,800,3)
m0 = first_obs.mean()
m1 = first_render.mean()
assert m0 > 0 and m0 < 255
assert abs(m0 - m1) < 5
# plt.imshow(first_obs)
# plt.pause(2)


# Check that the observation shapes match in reset and step 
# 检查reset函数和step函数的输出state是否保持尺寸一致
second_obs, _, _, _ = env.step(0)
assert first_obs.shape == env.observation_space.shape
assert first_obs.shape == second_obs.shape

# Test the PyTorch observation wrapper
env = PyTorchObsWrapper(env)
first_obs = env.reset()
second_obs, _, _, _ = env.step(0)
assert first_obs.shape == env.observation_space.shape
assert first_obs.shape == second_obs.shape

# Test TextFrame
# make sure it loads the TextFrame with no issues
class TestText(gym_miniworld.envs.threerooms.ThreeRooms):
    def _gen_world(self):
        super()._gen_world()
        self.entities.append(TextFrame(
            pos = [0, 1.35, 7],
            dir = math.pi/2,
            str = 'this is a test')
            )
env = TestText()
# env.render()
# plt.pause(3)

# Basic collision detection test
# Make sure the agent can never get outside of the room
# 基础碰撞功能检测，确保智能体不会走出房间
env = gym.make('MiniWorld-OneRoom-v0')
for _ in range(30):
    env.reset()
    room = env.rooms[0]
    for _ in range(30):
        env.step(env.actions.move_forward)
        x, _, z = env.agent.pos
        assert x >= room.min_x and x <= room.max_x
        assert z >= room.min_z and z <= room.max_z

# Try loading each of the available environments
for env_id in gym_miniworld.envs.env_ids:
    if 'RemoteBot' in env_id:
        continue

    print('Testing "' + env_id + '"')
    env = gym.make(env_id)
    env.domain_rand = True
    # Try multiple random restarts
    for _ in range(15):
        env.reset()
        assert not env.intersect(env.agent, env.agent.pos, env.agent.radius)
        # Perform multiple random actions
        for _ in range(0, 20):
            action = env.rand.int(0, env.action_space.n)
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()
