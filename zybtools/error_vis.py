import numpy as np
import open3d as o3d


pcd1 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/购买-5/points.ply")
# pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/gaussian_semantics/output/sat_pose_instance4/point_cloud/iteration_1000/point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/logs/save_offical&sigmod/购买5/offical/best_scene_filtered.ply")
pcd3 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/logs/save_offical&sigmod/购买5/apperance/scene_29000_filtered.ply")



# o3d.visualization.draw_geometries([pcd2])
pcd2.paint_uniform_color([0.5, 0.5, 0.5])  # 把所有点渲染为灰色（灰兔子）
pcd_tree = o3d.geometry.KDTreeFlann(pcd1)  # 建立KD树索引
point = pcd2.points[1200]  # 设置索引点
k = 4000  # 查询邻点数目
[k, idx, _] = pcd_tree.search_knn_vector_3d(point, k)  # K近邻搜索
# pcd1.colors[1200] = [1, 0, 0]  # 设置索引点为红色
np.asarray(pcd1.colors)[idx[1:], :] = [0, 1, 0]  # K邻域的点，渲染为绿色
o3d.visualization.draw_geometries([pcd1])
