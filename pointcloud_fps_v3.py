import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time
from numba import cuda

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

# step1
@cuda.jit
def getFarthestPointKernel(d_data, d_temp, ind, d_result, d_farthest_point, n):
    idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    i = idx

    if i < n:
        if i != ind:
            length = (d_data[0][ind] - d_data[0][i]) ** 2 + (d_data[1][ind] - d_data[1][i]) ** 2 + (d_data[2][ind] - d_data[2][i]) ** 2
            d_temp[i] = length
        cuda.syncthreads()
        cuda.atomic.max(d_result, 0, d_temp[i])

        if d_temp[i] == d_result[0]:
            d_farthest_point[0] = i
            d_temp[i] = -1.0
# step2
@cuda.jit
def getFarthestPointKernel2(d_data, d_temp, d_rest, d_select, d_step2_result, d_farthest_point, rest_lens, select_lens):
    idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    i = idx
    j = select_lens - 1

    if i < rest_lens:
        length = (d_data[0][d_rest[i]] - d_data[0][d_select[j]]) ** 2 + (d_data[1][d_rest[i]] - d_data[1][d_select[j]]) ** 2 + (d_data[2][d_rest[i]] - d_data[2][d_select[j]]) ** 2
        cuda.syncthreads()

        if length < d_temp[d_rest[i]]:
            d_temp[d_rest[i]] = length
        cuda.syncthreads()
        cuda.atomic.max(d_step2_result, 0, d_temp[d_rest[i]])

        if d_temp[d_rest[i]] == d_step2_result[0]:
            d_farthest_point[0] = d_rest[i]
            d_temp[d_rest[i]] = -1.0

# FarthestPointSampling
def farthestPointSampling(points, samples_num):
    select = []
    lens = len(points[0])
    rest = [num for num in range(0, lens)]
    max_dist = -1e10
    farthest_point = np.ones(1, dtype=np.int64)
    temp = np.zeros(lens, dtype=np.float64)
    result = np.zeros(1, dtype=np.float64)

    random.seed(1)
    ind = random.randint(0, lens)
    select.append(ind)
    rest.remove(ind)
    
    threadsperblock = 256
    blockspergrid = (lens + (threadsperblock - 1)) // threadsperblock
    d_data = cuda.to_device(points)
    d_temp = cuda.to_device(temp)
    d_result = cuda.to_device(result)
    d_farthest_point = cuda.to_device(farthest_point)
    getFarthestPointKernel[threadsperblock, blockspergrid](d_data, d_temp, ind, d_result, d_farthest_point, lens)
    cuda.synchronize()
    farthest_point = d_farthest_point.copy_to_host()

    select.append(farthest_point[0])
    rest.remove(farthest_point[0])

    while len(select) <  samples_num:
        
        rest_lens = len(rest)
        select_lens = len(select)
        step2_result = np.zeros(1, dtype=np.float64)
        threadsperblock = 256
        blockspergrid2 = (rest_lens + (threadsperblock - 1)) // threadsperblock
        d_rest = cuda.to_device(rest)
        d_select = cuda.to_device(select)
        d_step2_result = cuda.to_device(step2_result)

        getFarthestPointKernel2[threadsperblock, blockspergrid2](d_data, d_temp, d_rest, d_select, d_step2_result, d_farthest_point, rest_lens, select_lens)
        cuda.synchronize()
        farthest_point = d_farthest_point.copy_to_host()

        select.append(farthest_point[0])
        rest.remove(farthest_point[0])
    
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
    print("V3 版本")
    print("Sample点数 1000")
    start_gpu = time.time()
    sample_data = farthestPointSampling(data, 1000)
    end_gpu = time.time()

    print('------ Gpu process time:' + str(end_gpu - start_gpu))
    
    displayPoint(sample_data, "airplane")