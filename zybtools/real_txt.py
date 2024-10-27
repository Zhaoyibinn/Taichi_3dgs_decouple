import numpy as np
import json
from scipy.spatial.transform import Rotation as R


def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix

txt_path = "/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_2/2.txt"
saved_txt_path = "/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_2/sparse/0/images.txt"
json_path = "/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_2/transforms_origin.json"
saved_json_path = "/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_2/transforms.json"
pic_num = 0
all_xyz = []
all_q = []
with open(txt_path, 'r', encoding='utf-8') as file:
    for line in file:
        pic_num +=1
        data = line.split(" ")
        data = data[:-1]
        xyz = [float(data[-3]),float(data[-2]),float(data[-1])]
        xyz = np.array(xyz)
        all_xyz.append(xyz)
        q = [float(data[1]),float(data[2]),float(data[3]),float(data[0])]
        q = np.array(q)
        all_q.append(q)
        # print(data)
all_xyz = np.array(all_xyz)
all_q = np.array(all_q)
all_weizi = np.concatenate((all_q, all_xyz), axis=1).tolist()

# 打开一个文件用于写入
with open(saved_txt_path, 'w', encoding='utf-8') as file:
    # 遍历列表中的每个元素
    n = 1
    for item in all_weizi:
        # 写入元素到文件

        # q = item[:4]
        q = [item[3],item[0],item[1],item[2]]
        # t = item[4:]
        t = (np.array(item[4:]) * 1000).tolist()
        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()
        R_inv = np.linalg.inv(rotation_matrix)
        T = -np.dot(R_inv, t)
        r = R.from_matrix(R_inv)
        q_inv = r.as_quat()
        item[:4] = [q_inv[3],q_inv[0],q_inv[1],q_inv[2]]
        # item[:4] = [1,0,0,0]
        item[4:] = T
        # item[4:] = np.dot(R_inv, t)


        item = ' '.join(map(str, item))
        colmap_write = str(n) + " "+item+" 1 r_" + str(n) + ".png"
        file.write(colmap_write + '\n')  # 写入元素后跟一个换行符
        # 写入一个空行
        file.write('\n')
        n+=1


print("saved_txt")

# # 打开JSON文件并读取数据
# with open(json_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
#     pic_data = data['frames']
#     used_data  = pic_data[:pic_num]
# for i in range(pic_num):
#     q = all_q[i]
#     t = all_xyz[i]
#     r = R.from_quat(q)
#     rotation_matrix = r.as_matrix()
#     T = np.eye(4)
#     T[:3,:3] = rotation_matrix
#     T[:3,3] = t
#
#     used_data[i]['transform_matrix'] = T.tolist()
#     # used_data[i]['transform_matrix'] = list(q) + list(t)
#     data['frames'] = used_data
#
#
#     print("ok")
#
# # json_data = json.dumps(used_data)
#
# with open(saved_json_path, "w") as f:
#     json.dump(data, f)
#
# print("ok")