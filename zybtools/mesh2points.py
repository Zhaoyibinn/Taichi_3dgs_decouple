import time
import open3d as o3d
import numpy as np


path = "/media/zhaoyibin/common/3DRE/论文相关/paper/old_data/NeRF/nerf_synthetic/lego/lego.stl"
mesh = o3d.io.read_triangle_mesh(path)

output_cloud = "/media/zhaoyibin/common/3DRE/论文相关/paper/old_data/NeRF/nerf_synthetic/lego/points.ply"
# mesh.compute_vertex_normals()
# mesh.paint_uniform_color([0.9, 0.1, 0.1])




# 均匀采样5000个点
pcd = mesh.sample_points_uniformly(number_of_points=40000)
o3d.visualization.draw_geometries([pcd], window_name="pcd")

# 泊松采样5000个点，边缘点分布更加均匀，但是耗时更长
# pcd = mesh.sample_points_poisson_disk(number_of_points=5000, init_factor=10)
# o3d.visualization.draw_geometries([pcd], window_name="pcd")

o3d.io.write_point_cloud(output_cloud, pcd)