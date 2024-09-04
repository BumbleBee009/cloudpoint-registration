import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer
from pathlib import Path
import torch.utils.data
from plyfile import PlyData, PlyElement
import open3d as o3d
import cv2
import math
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from graphs import *
from provider import * 
from data_utils import *

def plyread(filepath, points_only=True):
    """Loads a ply file. """
    """convert from a ply file. include the label and the object number"""
    #---read the ply file--------
    plydata = PlyData.read(filepath)
    # print(filepath)
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
    cloudpoints_xyz =  torch.from_numpy(xyz).type(dtype=torch.float)

    return cloudpoints_xyz

def readcomponents(filepath):
    # 定义一个空列表来存储行数据
    data_list = []

    # 打开文件
    with open(filepath, 'r') as file:
        # 逐行读取文件
        for line in file:
            # 使用 strip() 去掉行末的换行符
            line = line.strip()
            
            # 将行按照空格分隔，转换为列表
            # 可以根据文件的实际情况调整分隔符
            line_list = line.split()
            
            # 根据需要，将列表中的元素转换为合适的数据类型
            # 在这个示例中，将每个元素转换为整数或浮点数
            # 您可以根据需求修改数据类型
            line_list = [int(value) for value in line_list]
            
            # 将行列表添加到数据列表中
            data_list.append(line_list)
        data_array = np.array(data_list, dtype='object')
        radix_sort(data_array)
        return data_array
    
def partition2array(xyz, components,visual = False, n = 1): # Identify points in the same cluster by color
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    # color = np.zeros(xyz.shape)
    color = np.zeros(len(xyz))
    # pcd = o3d.geometry.PointCloud()
    # for i_com in range(0, len(components)):
    #     color[components[i_com], :] = [random_color(), random_color()
    #     , random_color()]
    for i_com in range(0, len(components)):
        color[components[i_com]] = [random_color()]
    # prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    # , ('green', 'u1'), ('blue', 'u1')]
    # vertex_all = np.empty(len(xyz), dtype=prop)
    # if visual:
    #     for i in range(0, 3):
    #         vertex_all[prop[i][0]] = xyz[:, i]
    #     for i in range(0, 3):
    #         vertex_all[prop[i+3][0]] = color[:, i]

    if n > len(components):
        n = len(components)

    components_roi = []
    single_roi = []
    for i in range(0,n):
        components_roi = components_roi + components[i]
        singlecloud_roi = random_downsample(xyz[components[i]],0.1)
        single_roi.append(singlecloud_roi)
    color = color/255.0

    # # 将 NumPy 数组转换为 Open3D 点云格式
    # pcd.points = o3d.utility.Vector3dVector(xyz[components_roi])

    # # 将颜色数据转换为 Open3D 格式
    # pcd.colors = o3d.utility.Vector3dVector(color[components_roi])

    return single_roi, xyz[components_roi], color[components_roi]

def compute_overlap(pcd1,pcd2,threshold=0.05):
    # 创建KDTree
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)
    
    # 计算点云1中每个点到点云2的最近邻距离
    distances1, _ = tree1.query(pcd2, k=1)
    
    # 计算点云2中每个点到点云1的最近邻距离
    distances2, _ = tree2.query(pcd1, k=1)
    
    # 计算在阈值内的点的数量
    overlap_count1 = np.sum(distances1 < threshold)
    overlap_count2 = np.sum(distances2 < threshold)
    
    # 计算重叠点的比例
    total_points1 = len(pcd1)
    total_points2 = len(pcd2)
    
    overlap_ratio1 = overlap_count1 / total_points1
    overlap_ratio2 = overlap_count2 / total_points2
    
    # 平均重叠度
    average_overlap = (overlap_ratio1 + overlap_ratio2) / 2
    return average_overlap

def mse_similarity(pcd1, pcd2):
    # 创建KDTree
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)
    
    # 计算点云1中每个点到点云2的最近邻距离
    distances1, _ = tree1.query(pcd2, k=1)
    
    # 计算点云2中每个点到点云1的最近邻距离
    distances2, _ = tree2.query(pcd1, k=1)
    
    # 计算均方误差（MSE）
    mse = np.mean(distances1**2) + np.mean(distances2**2)
    return mse

def visualize_point_cloud(pcd):
    # 使用 Open3D 可视化函数
    o3d.visualization.draw_geometries([pcd])

def calculate_weighted_centroid(arrays):
    # 展开数组并统计每个元素的出现次数
    # flat_array = [item for sublist in arrays for item in sublist]
    flat_array = [len(sublist) for sublist in arrays]
    indices = np.arange(len(flat_array))
    indices = indices + 1

    # 计算重心
    numberator_step = np.abs(indices) * np.abs(flat_array)
    numerator = math.fsum(numberator_step)   # 计算乘积之和
    denominator = np.sum(flat_array)  # 计算元素数量的总和
    centroid = int(numerator) / denominator  # 计算重心
    rounded = round(centroid/2)
    # sum =  math.fsum(flat_array[:rounded])
    # print(sum)
    return 50

def dataloader(filepath,filetxt):
    xyz = plyread(filepath)
    components = readcomponents(filetxt)
    Threshold = calculate_weighted_centroid(components)
    single_roi, xyz_, color_ = partition2array(xyz, components,True,Threshold)
    return single_roi, xyz_, color_

def plot_color_point_cloud(xyz1,color1,xyz2,color2):
    sizes = 1
    fig = plt.figure(figsize=(14, 6))

    # 第一个三维子图
    ax = fig.add_subplot(121, projection='3d')
    # 绘制彩色三维点云
    sc1 = ax.scatter(xyz1[:,0], xyz1[:,1], xyz1[:,2], c=color1, s=1, alpha=0.5, cmap='viridis')
    # 添加标题和轴标签
    ax.set_title('Colorful 3D Scatter Plot 1')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # 第二个三维子图
    ay = fig.add_subplot(122, projection='3d')
    # 绘制彩色三维点云
    sc2 = ay.scatter(xyz2[:,0], xyz2[:,1], xyz2[:,2], c=color2, s=1, alpha=0.5, cmap='viridis')
    # 添加标题和轴标签
    ay.set_title('Colorful 3D Scatter Plot 2')
    ay.set_xlabel('X-axis')
    ay.set_ylabel('Y-axis')
    ay.set_zlabel('Z-axis')

    # 添加颜色条
    fig.colorbar(sc1, ax=ax)
    fig.colorbar(sc2, ax=ay)
    # 调整子图布局
    plt.tight_layout()

    # 显示图像
    plt.show(block=False)


def plot_point_cloud(points1,points2):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig2 = plt.figure()
    ay = fig2.add_subplot(111, projection='3d')
    ay.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=1)
    ay.set_xlabel('X')
    ay.set_ylabel('Y')
    ay.set_zlabel('Z')

    plt.show()

def random_downsample(point_cloud, sample_ratio):
    num_points = point_cloud.shape[0]
    sample_size = int(num_points * sample_ratio)
    indices = np.random.choice(num_points, sample_size, replace=False)
    return point_cloud[indices]

if __name__=="__main__":
    print('begin')
    
    filepath1 = 'L:/DataSet/GLR3d/GLR1.2/BLYM/train/BLYM_station14.ply'
    filetxt1 = 'L:/DataSet/GLR3d/GLR1.2/BLYM/train/BLYM_station14.txt'
    filepath2 = 'L:/DataSet/GLR3d/GLR1.2/BLYM/train/BLYM_station13.ply'
    filetxt2 = 'L:/DataSet/GLR3d/GLR1.2/BLYM/train/BLYM_station13.txt'
    single_roi1,xyz1,colors1 = dataloader(filepath1,filetxt1)
    print('computed first station')
    single_roi2,xyz2,colors2 = dataloader(filepath2,filetxt2)
    print('computed second station')
    
    plot_color_point_cloud(xyz1,colors1,xyz2,colors2)

    cv2.namedWindow('test')
    for i, values1 in enumerate(single_roi1):
        flag = True
        for j, values2 in enumerate(single_roi2):
            temp = mse_similarity(values1,values2)
            if flag: min_overlap = temp;  location = j; flag = False 
            if temp < min_overlap:
                min_overlap = temp
                location = j
            print(min_overlap)
        plot_point_cloud(single_roi1[i],single_roi2[location])
        cv2.waitKey(0)
        print('The '+ str(i) +'th')

    # visualize_point_cloud(pcd)
    print('success')


