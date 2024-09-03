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
    color = np.zeros(xyz.shape)
    pcd = o3d.geometry.PointCloud()
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color()
        , random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    , ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    if visual:
        for i in range(0, 3):
            vertex_all[prop[i][0]] = xyz[:, i]
        for i in range(0, 3):
            vertex_all[prop[i+3][0]] = color[:, i]

    if n > len(components):
        n = len(components)
    extracted_points_array = []
    extracted_points_all = []
    extracted_points_color = []
    for i in range(0,n):
        if visual:
            extracted_points1 = np.empty(vertex_all.shape)
            extracted_points1 = vertex_all[components[i]]
            extracted_points_all.extend(extracted_points1)

            extracted_points2 = np.empty(xyz.shape)
            extracted_points2 = xyz[components[i]]
            extracted_points_array.append(extracted_points2)

            extracted_points3 = np.empty(xyz.shape)
            extracted_points3 = color[components[i]]
            extracted_points_color.append(extracted_points2)
        else:
            extracted_points = np.empty(xyz.shape)
            extracted_points = xyz[components[i]]
            extracted_points_array.append(extracted_points)

    components_roi = []
    single_roi = []
    for i in range(0,n):
        components_roi = components_roi + components[i]
        single_roi[i] = xyz[components[i]]
    color = color/255.0

    # 将 NumPy 数组转换为 Open3D 点云格式
    pcd.points = o3d.utility.Vector3dVector(xyz[components_roi])

    # 将颜色数据转换为 Open3D 格式
    pcd.colors = o3d.utility.Vector3dVector(color[components_roi])

    return pcd
        
    # for i in range(0,len(components)):
    #     array = np.empty(shape=(len(components)),dtype = int)
    #     array[i] = len(components[i])
    #     np.savetxt('/root/code/GLR1.0/array.txt',array)
    
    # cloudpoints = extracted_points_array
    # if visual:
    #     now = datetime.now()
    #     formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    #     extracted_points_all_np = np.array(extracted_points_all)
    #     ply = PlyData([PlyElement.describe(extracted_points_all_np, 'vertex')], text=True)
    #     ply.write('/root/datasets/test/'+ formatted_time + '.ply')
    # return cloudpoints

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
    sum =  math.fsum(flat_array[:rounded])
    print(sum)
    return rounded

if __name__=="__main__":
    print('begin')
    
    filepath = 'L:/DataSet/GLR3d/GLR1.2/BLYM/train/BLYM_station12.ply'
    filetxt = 'L:/DataSet/GLR3d/GLR1.2/BLYM/train/BLYM_station12.txt'
    # filepath = 'L:/DataSet/GLR3d/GLR1.2/Basement/train/room_station2.ply'
    # filetxt = 'L:/DataSet/GLR3d/GLR1.2/Basement/train/room_station2.txt'
    xyz = plyread(filepath)
    components = readcomponents(filetxt)
    print(len(components))
    Threshold = calculate_weighted_centroid(components)
    # Threshold = 1
    # for i in range(0,len(components)):
    #     if len(components[i]) < 10000 or i == len(components)-1:
    #         Threshold = i
    #         break
    print(Threshold)
    pcd = partition2array(xyz, components,True,Threshold)
    visualize_point_cloud(pcd)
    print('success')


