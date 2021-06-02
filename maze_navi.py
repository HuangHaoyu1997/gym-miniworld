#!/usr/bin/env python3

import os
import numpy as np
import math
import gym
import gym_miniworld
from gym_miniworld.wrappers import PyTorchObsWrapper, GreyscaleWrapper
from gym_miniworld.entity import TextFrame
import matplotlib.pyplot as plt


# print(gym_miniworld.envs.env_ids)
'''
['MiniWorld-CollectHealth-v0', 'MiniWorld-FourRooms-v0',     'MiniWorld-Hallway-v0', 
'MiniWorld-Maze-v0',           'MiniWorld-MazeS2-v0',        'MiniWorld-MazeS3-v0', 
'MiniWorld-MazeS3Fast-v0',     'MiniWorld-OneRoom-v0',       'MiniWorld-OneRoomS6-v0', 
'MiniWorld-OneRoomS6Fast-v0',  'MiniWorld-PickupObjs-v0',    'MiniWorld-PutNext-v0', 
'MiniWorld-RemoteBot-v0',      'MiniWorld-RoomObjs-v0',      'MiniWorld-Sidewalk-v0', 
'MiniWorld-SimToRealGoTo-v0',  'MiniWorld-SimToRealPush-v0', 'MiniWorld-TMaze-v0', 
'MiniWorld-TMazeLeft-v0',      'MiniWorld-TMazeRight-v0',    'MiniWorld-ThreeRooms-v0', 
'MiniWorld-WallGap-v0',        'MiniWorld-YMaze-v0',         'MiniWorld-YMazeLeft-v0', 
'MiniWorld-YMazeRight-v0'] 
'''

def look_around(env):
    '''
    环顾360°收集全视角环境图像
    执行24次turn_left，每次转角15°
    '''
    s_buffer = []
    r_buffer = []
    done = False
    for i in range(24):
        obs,r,done,_ = env.step(env.actions.turn_left)
        s_buffer.append(obs)
        r_buffer.append(r)
    return s_buffer,r_buffer,done


env = gym.make('MiniWorld-FourRooms-v0')
obs = env.reset()
last_pos = env.agent.pos
obs,r,done = look_around(env)

while done is not True:
    env.render()
    top_img = env.render_top_view() # 俯视图
    # plt.imshow(img)
    # plt.pause(0.000000001)
    
    obs,r,done,_ = env.step(env.action_space.sample())
    print('pos:',env.agent.pos,'dir:',env.agent.dir/math.pi*180)
    delta_x = np.linalg.norm(last_pos-env.agent.pos)
    last_pos = env.agent.pos

    if delta_x > 0.1:
        look_around(env)

