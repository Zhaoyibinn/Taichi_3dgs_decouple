import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement

# all_path = "logs/save_offical&sigmod/B330/ours/"
# gt_pth = "data/B330/points.ply"
#
# parquet_path = all_path + "scene_instance.parquet"
# ply_path = all_path + "scene_instance.ply"
# saved_path = all_path + "scene_instance_trans.ply"


# ply_path = "logs/save_offical&sigmod/UDM/3DGS/scene_29000.ply"
# gt_pth = "data/UDM/points.ply"
# saved_path = "logs/save_offical&sigmod/UDM/3DGS/scene_29000_trans.ply"


ply_path = "logs/save_offical&sigmod/UDM/3DGS/scene_instance.ply"
gt_pth = "data/UDM/points.ply"
saved_path = "logs/save_offical&sigmod/UDM/3DGS/scene_instance_trans.ply"
plydata = PlyData.read(ply_path)
gt_pcd = o3d.io.read_point_cloud(gt_pth)
ours_pcd = o3d.geometry.PointCloud()


xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])), axis=1)

points_vector = o3d.utility.Vector3dVector(xyz)
ours_pcd.points = points_vector
try:
    instance =np.asarray(plydata.elements[0]["instance"])
    instance = (instance==0.5)
except:
    print("no instace")

dist = np.array(ours_pcd.compute_point_cloud_distance(gt_pcd))


f_dc  = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                np.asarray(plydata.elements[0]["f_dc_1"]),
                np.asarray(plydata.elements[0]["f_dc_2"])), axis=1)

for i in range(45):

    # idx = i + 1
    name = "f_rest_" + str(i)
    f_re_now = np.asarray(plydata.elements[0][name])
    try:
        f_re =  np.vstack((f_re,f_re_now))
    except:
        f_re = np.array([np.asarray(plydata.elements[0][name])])
f_rest = f_re.T
# f_sh = np.hstack((f_dc,f_re))
# valid_point_cloud = self.point_cloud[self.point_invalid_mask == 0].detach().cpu()
# valid_point_cloud_features = self.point_cloud_features[self.point_invalid_mask == 0].detach().cpu()
# xyz = valid_point_cloud.numpy()
normals = np.zeros_like(xyz)
# f_sh = f_sh.reshape(-1, 3, 16)
# f_dc = f_sh[..., 0]
# f_rest = f_sh[..., 1:].reshape(-1, 45)
opacities = np.asarray(plydata.elements[0]["opacity"]).reshape(-1,1)
scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                np.asarray(plydata.elements[0]["scale_1"]),
                np.asarray(plydata.elements[0]["scale_2"])), axis=1)
rotation = np.stack((np.asarray(plydata.elements[0]["rot_3"]),
                np.asarray(plydata.elements[0]["rot_0"]),
                np.asarray(plydata.elements[0]["rot_1"]),
                np.asarray(plydata.elements[0]["rot_2"])), axis=1)


ok_bool = dist<0.1



f_dc[ok_bool] = [-100,100,-100]
f_dc[~ok_bool] = [100,-100,-100]


# instance = instance[ok_bool]

f_rest = np.zeros_like(f_rest)

try:
    xyz = xyz[instance]
    f_dc = f_dc[instance]
    f_rest = f_rest[instance]
    normals = normals[instance]
    opacities = opacities[instance]
    scale = scale[instance]
    rotation = rotation[instance]
except:
    print("no instance")

opacities.fill(1)


# scale = scale * 1.5




def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(f_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

elements = np.empty(xyz.shape[0], dtype=dtype_full)
attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
elements[:] = list(map(tuple, attributes))
el = PlyElement.describe(elements, 'vertex')
PlyData([el]).write(saved_path)
print("end")