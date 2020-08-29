import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time


def readXYZfile(filename, Separator):
    data = [[], [], []]
    f = open(filename, 'r')
    line = f.readline()
    num = 0
    while line:  # 按行读入点云
        c, d, e, _, _, _ = line.split(Separator)
        data[0].append(c)  # x坐标
        data[1].append(d)  # y坐标
        data[2].append(e)  # z坐标
        num = num + 1
        line = f.readline()
    f.close()

    #string型转float型
    x = [float(data[0]) for data[0] in data[0]]
    z = [float(data[1]) for data[1] in data[1]]
    y = [float(data[2]) for data[2] in data[2]]
    print("读入点的个数为：{}个。".format(num))
    point = [x, y, z]
    return point

# 三维离散点图显示点云
def displayPoint(data, title):
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 点数量太多不予显示
    while len(data[0]) > 20000:
        print("点太多了！")
        exit()
    #散点图参数设置
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    ax.scatter3D(data[0], data[1], data[2], c = 'b', marker = '.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def displayPoint2(data,sample_data, title):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    while len(data[0]) > 20000:
        print("点太多了！")
        exit()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title(title)
    ax.scatter3D(data[0], data[1], data[2], c = 'b', marker = '.')
    ax.scatter3D(sample_data[0], sample_data[1], sample_data[2], c = 'r', marker = '.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


# FarthestPointSampling
def farthestPointSampling(points, samples_num):
    select = []
    lens = len(points[0])
    rest = [num for num in range(0, lens)]
    max_dist = -1e10
    farthest_point = 1e10
    temp = {}

    random.seed(1)
    ind = random.randint(0, lens)
    select.append(ind)
    rest.remove(ind)

    for i in range (lens):
        if i != ind:
            length = (points[0][ind] - points[0][i]) ** 2 + (points[1][ind] - points[1][i]) ** 2 + (points[2][ind] - points[2][i]) ** 2
            temp[i] = length
            if length > max_dist:
                max_dist = length
                farthest_point = i
    #print(farthest_point)
    select.append(farthest_point)
    rest.remove(farthest_point)

    while len(select) <  samples_num:
        min_length = []
        max_dist = -1e10

        for i in range(len(rest)):

            for j in range(len(select) - 1, len(select)):
                length = (points[0][rest[i]] - points[0][select[j]]) ** 2 + (points[1][rest[i]] - points[1][select[j]]) ** 2 + (points[2][rest[i]] - points[2][select[j]]) ** 2
                if length < temp[rest[i]]:
                    temp[rest[i]] = length
                    #print(min_dist)

            min_length.append((rest[i], temp[rest[i]]))

            if list(min_length[i])[1] > max_dist:
                max_dist = list(min_length[i])[1] 
                farthest_point = list(min_length[i])[0]

        select.append(farthest_point)
        rest.remove(farthest_point)

    new_x = []
    new_y = []
    new_z = []
    for i in range(len(select)):
        new_x.append(points[0][select[i]])
        new_y.append(points[1][select[i]])
        new_z.append(points[2][select[i]])

    point = [new_x, new_y, new_z]

    return point 

if __name__ == "__main__":
    data = readXYZfile("airplane_0032.txt", ',')
    #displayPoint(data, "airplane")
    print("V2 版本")
    print("Sample点数 1000")
    start_cpu = time.time()
    sample_data = farthestPointSampling(data, 1000)
    end_cpu = time.time()

    print('------ cpu process time:' + str(end_cpu - start_cpu))
    #print(sample_data)

    #displayPoint2(data, sample_data, "airplane")
    displayPoint(sample_data, "airplane")