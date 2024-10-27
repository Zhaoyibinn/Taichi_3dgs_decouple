import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  plotnine import *
import seaborn as sns
import open3d as o3d


all_data = []
shangjie = 100
xiajie = 0.001
stm_beishu  = 10

# 使用pandas读取CSV文件
with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买4/csv/3DGS.txt', 'r') as file:
    content = file.read()
GS = []
for i in content.split("\n"):
    if i =='':
        continue
    if i[0]=='/' :
        continue
    if float(i.split()[-4])>shangjie:
        GS.append(shangjie)
        continue
    if float(i.split()[-4])<xiajie:
        GS.append(xiajie)
        continue
    GS.append(float(i.split()[-4]))
GS = np.array(GS)
all_data.append(GS)





with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买4/csv/colmap.txt', 'r') as file:
    content = file.read()
colmap = []
for i in content.split("\n"):
    if i =='':
        continue
    if i[0]=='/' :
        continue
    if float(i.split()[-4])>shangjie:
        colmap.append(shangjie)
        continue
    if float(i.split()[-4])<xiajie:
        colmap.append(xiajie)
        continue
    colmap.append(float(i.split()[-4]))
colmap = np.array(colmap)
all_data.append(colmap)






with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买4/csv/scene_instance_less_filtered.txt', 'r') as file:
    content = file.read()
ours = []
for i in content.split("\n"):
    if i =='':
        continue
    if i[0]=='/' :
        continue
    if float(i.split()[3])>shangjie:
        ours.append(shangjie)
        continue
    if float(i.split()[3])<xiajie:
        ours.append(xiajie)
        continue
    ours.append(float(i.split()[3]))
ours = np.array(ours)
all_data.append(ours)




with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买4/csv/2DGS.txt', 'r') as file:
    content = file.read()
GS2D = []
for i in content.split("\n"):
    if i =='':
        continue
    if i[0]=='/' :
        continue
    if float(i.split()[-4])>shangjie:
        GS2D.append(shangjie)
        continue
    if float(i.split()[-4])<xiajie:
        GS2D.append(xiajie)
        continue
    GS2D.append(float(i.split()[-4]))
GS2D = np.array(GS2D)
all_data.append(GS2D)
all_bad_points = []
for add in np.linspace(0, 1, 1000):
    bad_points_cat = []
    Q3 = GS.mean() + stm_beishu * GS.std()
    Q3 = add
    bad_points = np.sum(GS < Q3)/GS.shape[0]
    bad_points_cat.append(bad_points)
    # print("GS:",bad_points)

    Q3 = colmap.mean() +stm_beishu * colmap.std()
    Q3 = add
    bad_points = np.sum(colmap < Q3)/colmap.shape[0]
    bad_points_cat.append(bad_points)
    # print("colmap:",bad_points)

    Q3 = ours.mean() + stm_beishu * ours.std()
    Q3 = add
    bad_points = np.sum(ours < Q3)/ours.shape[0]
    bad_points_cat.append(bad_points)
    # print("ours:",bad_points)

    Q3 = GS2D.mean() + stm_beishu * GS2D.std()
    Q3 = add
    bad_points = np.sum(GS2D < Q3)/GS2D.shape[0]
    bad_points_cat.append(bad_points)
    # print("2DGS:",bad_points)
    all_bad_points.append(bad_points_cat)

all_bad_points = np.array(all_bad_points).T




fig, ax = plt.subplots(figsize=(15, 8))
x = np.linspace(0, 1, 1000)
ax.plot(x, all_bad_points[0], label='3DGS')
ax.plot(x, all_bad_points[1], label='Colmap')
ax.plot(x, all_bad_points[3], label='2DGS')
ax.plot(x, all_bad_points[2], label='Ours')
# ax.legend(fontsize = 15)
ax.set_xlabel('CD Error Thershold',fontsize = 20)
ax.set_ylabel('Percentage',fontsize = 20)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# ax.set_yscale('log')
plt.show()

print("end")













