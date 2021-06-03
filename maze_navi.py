#!/usr/bin/env python3

import os
import numpy as np
import math
import gym
import gym_miniworld
from gym_miniworld.wrappers import PyTorchObsWrapper, GreyscaleWrapper
from gym_miniworld.entity import TextFrame
import matplotlib.pyplot as plt
import pickle

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
EPI_NUM = 2

def look_around(env):
    '''
    环顾360°收集全视角环境图像
    执行24次turn_left，每次转角15°
    '''
    s_buffer = []
    r_buffer = []
    done = False
    agent_pos = env.agent.pos
    for i in range(24):
        obs,r,done,_ = env.step(env.actions.turn_left)
        s_buffer.append(obs)
        r_buffer.append(r)
    return s_buffer,r_buffer,done,agent_pos

def gen_data():
    env = gym.make('MiniWorld-FourRooms-v0')
    for epi in range(EPI_NUM):
        data_set = [] # 收集训练数据，1条数据包括 1)同一位置的24个视角的图像，2)位置坐标
        obs = env.reset()
        last_pos = env.agent.pos # 记录坐标
        obs,r,done,pos = look_around(env) # 环顾四周

        while done is not True:
            env.render()
            top_img = env.render_top_view() # 俯视图
            # plt.imshow(img)
            # plt.pause(0.000000001)
            
            action = env.action_space.sample()
            obs,r,done,_ = env.step(action)
            if done: break
            if action == 2 or action == 3: 
                # 左右转向(0,1)是原地动作，不执行look_around函数
                # 前进后退(2,3)改变位置坐标，执行look_around函数
                obs,r,done,pos = look_around(env)
                delta_x = np.linalg.norm(last_pos - pos) # 每一步行进的距离
                if delta_x > 0.1: data_set.append([obs,pos]) # 发生位移才收集数据

            # agent_dir = env.agent.dir/math.pi*180 # 视角，按度计算
            last_pos = env.agent.pos # 更新位置坐标
            print(len(data_set))

        with open(str(epi)+'.pkl','wb') as f:
            pickle.dump(data_set,f)

def process_data(dir):
    with open(dir,'rb') as f:
        data = pickle.load(f)
    
    img,pos = [],[]
    for i in range(len(data)):
        img.append(data[i][0])
        pos.append(data[i][1])
    return img,pos

if __name__ == "__main__":
    gen_data()
    '''
    img, pos = process_data('D:/data/1.pkl')
    print(np.array(img).shape,np.array(pos).shape)
    pos_ = pos[0]
    for i in range(len(pos)):
        delta_x = np.linalg.norm(pos_-pos[i])
        pos_ = pos[i]
        print(delta_x)
    '''
