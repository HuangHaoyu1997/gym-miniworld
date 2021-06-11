import numpy as np
import random

def maze_map(map_size, room_size):
    gridtype = np.dtype([('cla', int), ('index', int)])
    maze_map = np.zeros((map_size, map_size), dtype=gridtype)
    num_room_row = int(map_size / room_size)
    for i in range(num_room_row):
        for j in range(map_size):
            maze_map[i * room_size][j][0] = 5
            maze_map[j][i * room_size][0] = 5
    # 为每个房间加不同数量的landmark
    landmarks=[3,7,8,9,10]
    for i in range(num_room_row):
        for j in range(num_room_row):
            landmark_size = 10
            # 为每个小房间里面加landmarks
            aa=random.randint(0, 4)
            print(aa)
            type = landmarks[aa]
            #type=3
            num_landmarks = random.randint(1, int(landmark_size * landmark_size/2))
            dis = gen_landmark_dis(landmark_size, num_landmarks, type)
            for p in range(landmark_size):
                for q in range(landmark_size):
                    maze_map[i * room_size + 10 + p][j * room_size + 10 + q][0] += dis[p][q]

    # 为每个房间加入几个门（在围墙上）
    # 这个工作暂时交给excel来做吧...目前实现的是每两个房间有50%的可能性是有门的
    for i in range(num_room_row):
        for j in range(num_room_row):
            flag = random.randint(0, 2)
            if flag:
                # 6是过道？
                for m in range(10):
                    maze_map[i * room_size][j * room_size + m+10][0] = 6
            flag = random.randint(0, 2)
            if flag:
                for m in range(10):
                    maze_map[i * room_size + m+10][j * room_size][0] = 6
    #加上上和左的围墙
    for i in range(map_size):
        maze_map[0][i][0]=5
        maze_map[i][0][0]=5

    return maze_map


def gen_landmark_dis(size, num, type):
    # size是指landmark聚集的小正方形的size
    dis = np.zeros((size, size), dtype=int)
    rd = random.sample(range(0, size * size), num)
    for i in range(num):
        r = int(rd[i] / size)
        c = int(rd[i] % size)
        dis[r][c] = type
    return dis


if __name__ == "__main__":

    random.seed(1000)
    mp = maze_map(300, 30)
    map = []
    for i in range(300):
        map.append([])
        for j in range(300):
            map[i].append(mp[i][j][0])
	print("这个map就是了，random seed不要改")
    # 5 墙
    # 0 地块
    # 6 过道
    # 7,8,9,10，landmark