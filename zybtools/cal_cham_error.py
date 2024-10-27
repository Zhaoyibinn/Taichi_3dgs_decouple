import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from chamferdist import ChamferDistance
import time

def o3d_vis(proj_points):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    proj_points_pcd = o3d.geometry.PointCloud()
    proj_points_pcd.points = o3d.utility.Vector3dVector(proj_points)
    o3d.visualization.draw_geometries([proj_points_pcd])
    return 0
def cal_error(points0,points1):
    p1 = torch.tensor([points0]).cuda()
    p2 = torch.tensor([points1]).cuda()

    # p1 = torch.tensor(points0).cuda()
    # p2 = torch.tensor(points1).cuda()

    # p1 = torch.randn(1, 100, 3).cuda()
    # p2 = torch.randn(1, 50, 3).cuda()
    s = time.time()
    chamferDist = ChamferDistance()
    dist_forward = chamferDist(p1, p2)
    dist_backward = chamferDist(p2, p1)
    error_forward = 0
    error_backward = 0
    # error_forward = dist_forward.detach().cpu().item()/np.array(points0).shape[0]
    error_backward = dist_backward.detach().cpu().item()/np.array(points1).shape[0]
    # error_forward = dist_forward.detach().cpu().item()
    # error_backward = dist_backward.detach().cpu().item()
    print(0.5 * error_forward + 0.5 * error_backward)
    # print(f"Time: {time.time() - s} seconds")
    return 0.5 * error_forward + 0.5 * error_backward

offical =[]
zyb = []
baifenbi = []

# 读取两个点云
pcd1 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买5/gt.ply")
# pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/gaussian_semantics/output/sat_pose_instance4/point_cloud/iteration_1000/point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买5/3DGS.ply")
pcd3 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买5/apperance.ply")
pcd4 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买5/colmap.ply")

# o3d.visualization.draw_geometries([pcd1,pcd2])
print("购买5")
# offical_error = cal_error(np.array(pcd1.points)*100,np.array(pcd2.points)*100)
# zyb_error = cal_error(np.array(pcd1.points)*100,np.array(pcd3.points)*100)
# colmap_error = cal_error(np.array(pcd1.points)*100,np.array(pcd4.points)*100)
offical_error = cal_error(pcd1.points,pcd2.points)
zyb_error = cal_error(pcd1.points,pcd3.points)
colmap_error = cal_error(pcd1.points,pcd4.points)

# cal_error(pcd1.points,pcd4.points)
offical.append(offical_error)
zyb.append(zyb_error)
baifenbi.append(zyb_error/offical_error)


pcd1 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/自建/gt.ply")
# pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/gaussian_semantics/output/sat_pose_instance4/point_cloud/iteration_1000/point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/自建/3DGS.ply")
pcd3 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/自建/scene_instance_less_filtered.ply")
pcd4 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/自建/colmap.ply")

# o3d.visualization.draw_geometries([pcd1,pcd2])
print("自建")
offical_error = cal_error(pcd1.points,pcd2.points)
zyb_error = cal_error(pcd1.points,pcd3.points)
colmap_error = cal_error(pcd1.points,pcd4.points)


# cal_error(pcd1.points,pcd4.points)
offical.append(offical_error)
zyb.append(zyb_error)
baifenbi.append(zyb_error/offical_error)

pcd1 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S1/gt.ply")
# pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/gaussian_semantics/output/sat_pose_instance4/point_cloud/iteration_1000/point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S1/3DGS_filtered.ply")
pcd3 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S1/scene_instance_less_filtered.ply")
pcd4 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S1/colmap_filtered.ply")

# o3d.visualization.draw_geometries([pcd1,pcd2])
print("S1")
offical_error = cal_error(pcd1.points,pcd2.points)
zyb_error = cal_error(pcd1.points,pcd3.points)
colmap_error = cal_error(pcd1.points,pcd4.points)
# cal_error(pcd1.points,pcd4.points)
offical.append(offical_error)
zyb.append(zyb_error)
baifenbi.append(zyb_error/offical_error)

pcd1 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S3_TANGO/gt.ply")
# pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/gaussian_semantics/output/sat_pose_instance4/point_cloud/iteration_1000/point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S3_TANGO/3DGS.ply")
pcd3 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S3_TANGO/scene_instance_less_filtered.ply")
pcd4 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S3_TANGO/colmap.ply")

# o3d.visualization.draw_geometries([pcd1,pcd2])
print("S3_TANGO")
offical_error = cal_error(pcd1.points,pcd2.points)
zyb_error = cal_error(pcd1.points,pcd3.points)
colmap_error = cal_error(pcd1.points,pcd4.points)

# cal_error(pcd1.points,pcd4.points)
offical.append(offical_error)
zyb.append(zyb_error)
baifenbi.append(zyb_error/offical_error)

pcd1 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/gt.ply")
# pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/gaussian_semantics/output/sat_pose_instance4/point_cloud/iteration_1000/point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/3DGS.ply")
pcd3 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/apperance.ply")
# o3d.visualization.draw_geometries([pcd1,pcd2])
pcd4 = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/colmap.ply")

print("S4_神舟")
offical_error = cal_error(pcd1.points,pcd2.points)
zyb_error = cal_error(pcd1.points,pcd3.points)
colmap_error = cal_error(pcd1.points,pcd4.points)
# cal_error(pcd1.points,pcd4.points)
offical.append(offical_error)
zyb.append(zyb_error)
baifenbi.append(zyb_error/offical_error)

x = [1,2,3,4,5]
# plt.plot(x, zyb, label='zyb')  # 第一条线，label用于图例
# plt.plot(x, offical, label='offical')  # 第二条线
plt.plot(x, baifenbi, label='zyb')
plt.plot(x, [1,1,1,1,1], label='offical')
plt.show()