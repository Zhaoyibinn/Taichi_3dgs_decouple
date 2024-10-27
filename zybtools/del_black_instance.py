
import os
import numpy as np
from click import clear
from plyfile import PlyData
import open3d as o3d

wenjian_path = "/home/zhaoyibin/3DRE/3DGS/taichi_semantics_splatting/logs/tat_train_experiment_downsample_warmup"

path = wenjian_path + "/scene_instance.ply"

plydata = PlyData.read(path)


xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])), axis=1)
instance =np.asarray(plydata.elements[0]["instance"])

xyz_instance = []
for i in range(instance.shape[0]):
    if instance[i] == 0.5:
        xyz_instance.append(xyz[i])

xyz_instance = np.array(xyz_instance)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(xyz_instance)

o3d.io.write_point_cloud(wenjian_path + "/scene_instance_less.ply", cloud)

print("end")