import os
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import math
from plyfile import PlyData, PlyElement
import torch
# import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import glob
import six

# Step 1: 提取点云的FPFH特征
def extract_fpfh_features(pcd, voxel_size=0.05):
    # 下采样点云
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # 计算法线
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    # 计算FPFH特征
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

# Step 2: 构建词典（通过KMeans聚类）
def build_vocabulary(fpfh_list, vocab_size=50):
    # 聚合所有点云的FPFH特征
    all_features = np.vstack([fpfh.data.T for fpfh in fpfh_list])
    # 使用KMeans聚类生成词典
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    kmeans.fit(all_features)
    return kmeans.cluster_centers_

# Step 3: 将点云特征转换为词袋直方图
# def point_cloud_to_bow(fpfh, vocabulary):
#     kmeans = KMeans(n_clusters=len(vocabulary), init=vocabulary, n_init=1)
#     labels = kmeans.predict(fpfh.data.T)
#     bow_histogram, _ = np.histogram(labels, bins=len(vocabulary))
#     return bow_histogram

def point_cloud_to_bow(fpfh, vocabulary):
    # 对于特征进行量化，找到最接近的聚类中心
    from sklearn.neighbors import NearestNeighbors

    # 使用 NearestNeighbors 查找最近的视觉单词（簇中心）
    nbrs = NearestNeighbors(n_neighbors=1).fit(vocabulary)
    distances, indices = nbrs.kneighbors(fpfh.data.T)

    # 生成词袋直方图
    bow_histogram, _ = np.histogram(indices, bins=len(vocabulary))
    return bow_histogram


# Step 4: 计算两个点云的词袋直方图相似性
def compute_similarity(hist1, hist2):
    return cosine_similarity(hist1.reshape(1, -1), hist2.reshape(1, -1))[0][0]


#//////////////////////////////////////////////////

def dataloader(filepath,filetxt):
    xyz = plyread(filepath)
    components = readcomponents(filetxt)
    Threshold = calculate_weighted_centroid(components)
    single_roi = partition2array(xyz, components,True,1,Threshold)
    return single_roi

def dataloader_single(filepath):
    xyz = plyread(filepath)
    filetemple = Path(filepath)
    filetxt = filetemple.with_suffix('.txt')
    components = readcomponents(filetxt)
    Threshold = calculate_weighted_centroid(components)
    single_roi = partition2array(xyz, components,False,1,Threshold)
    return single_roi

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

def radix_sort(components):
    max_num = 0
    for i in range(0,len(components)):
        if(max_num < len(components[i])):
            max_num = len(components[i])
    # print(max_num)
    exp = 1
    while max_num // exp >0:
        n = len(components)
        output = np.empty((n,), dtype=object)
        count = [0] * 10

        # 统计每个位上数字出现的次数
        for i in range(n):
            index = len(components[i]) // exp
            count[index % 10] += 1

        # 计算累计次数
        for i in range(1, 10):
            # count[i] += count[i - 1]
            count[10-i-1] += count[10-i]

        # 根据位数排序
        i = n - 1
        while i >= 0:
            index = len(components[i]) // exp
            output[count[index % 10] - 1] = components[i]
            count[index % 10] -= 1
            i -= 1

        # 将排序后的结果复制回原数组
        for i in range(n):
            components[i] = output[i]
        
        exp *= 10

def calculate_weighted_centroid(arrays):
    rounded = 0

    # 展开数组并统计每个元素的出现次数
    # flat_array = [item for sublist in arrays for item in sublist]
    flat_array = [len(sublist) for sublist in arrays]
    indices = np.arange(len(flat_array))
    indices = indices + 1
    if False:
        # 计算重心
        numberator_step = np.abs(indices) * np.abs(flat_array)
        numerator = math.fsum(numberator_step)   # 计算乘积之和
        denominator = np.sum(flat_array)  # 计算元素数量的总和
        centroid = int(numerator) / denominator  # 计算重心
        rounded = round(centroid/2)
        # sum =  math.fsum(flat_array[:rounded])
        # print(sum)
    else:
        sum =  math.fsum(flat_array)
        temp_sum = 0
        for i,value in enumerate(flat_array):
            temp_sum = temp_sum + flat_array[i]
            if temp_sum > sum/2:
                rounded = i
                break
    print('the kinds of pointblock is: ',rounded)
    return rounded

def fibonacci_iterative(n):
    a,b = 0,1
    for _ in range(n):
        yield a #使用yield可以逐个返回非伯纳切数字
        a,b=b,a+b

def partition2array(xyz, components,visual = False, type = 0, n = 1): # Identify points in the same cluster by color
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    pcd = o3d.geometry.PointCloud()
    colors = np.zeros(xyz.shape)
    # color = np.zeros(len(xyz))
    for i_com in range(0, len(components)):
        # color[components[i_com]] = [random_color()]
        colors[components[i_com], :] = [random_color(), random_color()
            , random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    , ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    if visual:
        for i in range(0, 3):
            vertex_all[prop[i][0]] = xyz[:, i]
        for i in range(0, 3):
            vertex_all[prop[i+3][0]] = colors[:, i]

    if n > len(components):
        n = len(components)

    components_roi = []
    extracted_points_all = []
    extracted_points_array = []
    for i in range(0,n):
        components_roi = components_roi + components[i]
        if visual:
            extracted_points1 = np.empty(vertex_all.shape)
            extracted_points1 = vertex_all[components[i]]
            extracted_points_all.extend(extracted_points1)
        else:
            extracted_points = np.empty(xyz.shape)
            extracted_points = xyz[components[i]]
            extracted_points_array.append(extracted_points)

    if type == 1:
        # 将 NumPy 数组转换为 Open3D 点云格式
        if len(xyz[components_roi]) > 10000:
            pcd.points = o3d.utility.Vector3dVector(xyz[components_roi])
        elif len(xyz) > 10000 or len(xyz) == 10000:
            print("don't use components")
            pcd.points = o3d.utility.Vector3dVector(xyz)
        elif len(xyz) < 10000:
            print("the number of points is less 1W!!!")
            pcd.points = o3d.utility.Vector3dVector(xyz)
        if visual:
            # 将颜色数据转换为 Open3D 格式
            pcd.colors = o3d.utility.Vector3dVector(colors[components_roi]/255.0)
        single_cloud = xyz

        now = datetime.now()
        for i in fibonacci_iterative(20):    
            voxel_size = 0.6/(i+1)
            downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            if len(downsampled_pcd.points) > 10000:
                break
        print('the number of points is:',len(downsampled_pcd.points))
        points_np = np.asarray(downsampled_pcd.points)
        points_tensor = torch.tensor(points_np, dtype=torch.float32)
        single_cloud = points_tensor
        if visual:
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            o3d.io.write_point_cloud('/home/lwh/result/downsample_result/'+ formatted_time + '_o.ply', pcd)
            o3d.io.write_point_cloud('/home/lwh/result/downsample_result/'+ formatted_time + '.ply', downsampled_pcd)
    elif type == 2:
        now = datetime.now()
        extracted_points_all_np = np.array(extracted_points_all)
        extracted_points_all_np_temp = random_downsample(extracted_points_all_np,0.01)
        single_cloud = extracted_points_all_np_temp
        if visual:    
            # formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            ply = PlyData([PlyElement.describe(extracted_points_all_np_temp, 'vertex')], text=True)
            ply.write('/root/code/test/'+ formatted_time + '.ply')
    elif type == 3:
        # # 将 NumPy 数组转换为 Open3D 点云格式
        pcd.points = o3d.utility.Vector3dVector(xyz[components_roi])
        # # 将颜色数据转换为 Open3D 格式
        pcd.colors = o3d.utility.Vector3dVector(colors[components_roi])

        # 设置ISS关键点提取的参数
        iss_keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
            salient_radius=0.005,  # 基于模型大小定义的显著性半径
            non_max_radius=0.005,  # 非极大值抑制半径
            gamma_21=0.975,        # 控制关键点的形状
            gamma_32=0.975,        # 控制关键点的形状
            min_neighbors=5        # 定义最小近邻数以识别关键点
        )
        single_cloud = iss_keypoints
        # 保存关键点到文件
        if visual: 
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            o3d.io.write_point_cloud('/root/code/test/'+ formatted_time + '.ply', iss_keypoints)

    return single_cloud

def random_downsample(point_cloud, sample_ratio):
    num_points = point_cloud.shape[0]
    sample_size = int(num_points * sample_ratio)
    indices = np.random.choice(num_points, sample_size, replace=False)
    return point_cloud[indices]

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

def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """

    if isinstance(ptns, six.string_types):
            ptns = [ptns]
    root = os.path.expanduser(root)
    samples = []

    for target in sorted(os.listdir(root)):
    # for target in os.listdir(root):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue

        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue

        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
            # for path in names:
                item = (path, target_idx)
                samples.append(item)
                
    return samples

def find_classes(root):
    #  find ${root}/${class}/* 
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# 测试代码
if __name__ == "__main__":

    # 加载点云数据
    # filepath1 = '/home/lwh/dataset/GLR/sg27_station/train/sg27_station5_intensity_rgb.ply'
    # filetxt1 = '/home/lwh/dataset/GLR/sg27_station/train/sg27_station5_intensity_rgb.txt'
    # filepath2 = '/home/lwh/dataset/GLR/BLYM/train/BLYM_station12.ply'
    # filetxt2 = '/home/lwh/dataset/GLR/BLYM/train/BLYM_station12.txt'
    rootdir = '/home/lwh/dataset/GLR'
    pattern = 'test/*.ply'
    classes, class_to_idx = find_classes(rootdir)
    samples = glob_dataset(rootdir, class_to_idx, pattern)
    for i in range(1,len(samples)):
        print('ID: ',i)
        print('Path: ', samples[i][0])
        dataloader_single(samples[i][0])
    # single_roi1 = dataloader(filepath1,filetxt1)
    # print('computed first station')
    # single_roi2 = dataloader(filepath2,filetxt2)
    # print('computed second station')


    # # Step 1: 提取FPFH特征
    # fpfh1 = [] ; fpfh2 = []
    # for i , values in enumerate(single_roi1):
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(values)
    #     fpfh1.append(extract_fpfh_features(pcd))
    
    # for i , values in enumerate(single_roi2):
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(values)
    #     fpfh2.append(extract_fpfh_features(pcd))

    # # Step 2: 生成词典
    # fpfh = fpfh1 + fpfh2
    # vocabulary = build_vocabulary(fpfh, vocab_size=100)

    # # Step 3: 将点云特征转换为词袋直方图
    # bow1 = [] ; bow2 = [] 
    # for i, values in enumerate(fpfh1):
    #     bow1.append(point_cloud_to_bow(fpfh1[i], vocabulary))
    # for i, values in enumerate(fpfh2):
    #     bow2.append(point_cloud_to_bow(fpfh2[i], vocabulary))

    # # Step 4: 计算点云之间的相似性
    # # sim12 = compute_similarity(bow1, bow2)
    # # cv2.namedWindow('test')
    # for i, values1 in enumerate(single_roi1):
    #     flag = True
    #     for j, values2 in enumerate(single_roi2):
    #         temp = compute_similarity(bow1[i], bow2[j])
    #         if flag: min_overlap = temp;  location = j; flag = False 
    #         if temp < min_overlap:
    #             min_overlap = temp
    #             location = j
    #         print(min_overlap)
    #     plot_point_cloud(single_roi1[i],single_roi2[location])
    #     # cv2.waitKey(0)
    #     print('The '+ str(i) +'th')
