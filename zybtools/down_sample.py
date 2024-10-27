import open3d as o3d

input = "/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/ply/scene_instance_less.ply"
output = "/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/ply/scene_instance_less_downsample.ply"
pcd = o3d.io.read_point_cloud(input)
print(pcd.points)
cl = pcd.voxel_down_sample(voxel_size=0.008)
print(cl.points)
o3d.io.write_point_cloud(output, cl)