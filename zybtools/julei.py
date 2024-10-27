import open3d as o3d
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt





pcd_path = "/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/real_camera/ours.ply"
out_pcd_path = "/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/real_camera/ours_less.ply"
# pcd_path = "/media/zhaoyibin/common/3DRE/PointCloud/probreg/data/ours.ply"
# out_pcd_path = "/media/zhaoyibin/common/3DRE/PointCloud/probreg/data/ours_less.ply"

pcd = o3d.io.read_point_cloud(pcd_path)
pcd = o3d.geometry.PointCloud(pcd)




nb_neighbors = 200
std_ratio = 1.0
print(pcd.points)
cl, ind = pcd.remove_statistical_outlier(
    nb_neighbors=nb_neighbors,
    std_ratio=std_ratio
)
pcd = pcd.select_by_index(ind)
print(pcd.points)

voxel_size = 0.01  # 体素大小为0.1x0.1x0.1

# 执行体素下采样
# pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(pcd.points)

o3d.io.write_point_cloud(out_pcd_path, pcd)
print("end")