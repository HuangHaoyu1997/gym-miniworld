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
['MiniWorld-CollectHealth-v0', 
'MiniWorld-FourRooms-v0', 
'MiniWorld-Hallway-v0', 
'MiniWorld-Maze-v0', 
'MiniWorld-MazeS2-v0', 
'MiniWorld-MazeS3-v0', 
'MiniWorld-MazeS3Fast-v0', 
'MiniWorld-OneRoom-v0', 
'MiniWorld-OneRoomS6-v0', 
'MiniWorld-OneRoomS6Fast-v0', 
'MiniWorld-PickupObjs-v0', 
'MiniWorld-PutNext-v0', 
'MiniWorld-RemoteBot-v0', 
'MiniWorld-RoomObjs-v0', 
'MiniWorld-Sidewalk-v0', 
'MiniWorld-SimToRealGoTo-v0', 
'MiniWorld-SimToRealPush-v0', 
'MiniWorld-TMaze-v0', 
'MiniWorld-TMazeLeft-v0', 
'MiniWorld-TMazeRight-v0', 
'MiniWorld-ThreeRooms-v0', 
'MiniWorld-WallGap-v0', 
'MiniWorld-YMaze-v0', 
'MiniWorld-YMazeLeft-v0', 
'MiniWorld-YMazeRight-v0'] 
'''
env = gym.make('MiniWorld-Maze-v0')
obs = env.reset()
done = False
while done is not True:
    env.render()
    obs,r,done,_ = env.step(env.action_space.sample())
    print(r)
