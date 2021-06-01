import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Ball, Barrier, Box, ImageFrame, Key, Office_chair, Office_desk, Potion, TextFrame

class FourRooms(MiniWorldEnv):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=2500,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1 , max_z=7
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-1
        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        # 放置多个目标
        self.box_red = self.place_entity(Box(color='red'))
        self.key_blue = self.place_entity(Key(color='blue'))
        self.ball_green = self.place_entity(Ball(color='green'))
        self.box_purple = self.place_entity(Box(color='purple'))
        self.box_snow = self.place_entity(Box(color='snow'))
        self.office_desk = self.place_entity(Office_desk())
        self.office_chair = self.place_entity(Office_chair())
        self.barrier = self.place_entity(Barrier())
        self.potion = self.place_entity(Potion())

        self.img_wall = self.place_entity(TextFrame(pos=[-5,0.5,5],dir=0,str='grass1'),room=0)

        print('pos_red,',self.box_red.pos)

        self.place_agent(room=0)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box_red):
            reward += self._reward()
            done = True

        return obs, reward, done, info
