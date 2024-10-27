import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  plotnine import *
import seaborn as sns
import open3d as o3d


all_data = []
shangjie = 1
xiajie = 0.001
stm_beishu  = 10
# 使用pandas读取CSV文件
with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买3/csv/3DGS.txt', 'r') as file:
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
print("GS error：",GS.mean(),"GS std：",GS.std())
all_data.append(GS)


with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买3/csv/colmap.txt', 'r') as file:
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
print("colmap error：",colmap.mean(),"colmap std：",colmap.std())
all_data.append(colmap)



with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买3/csv/scene_instance_less_filtered.txt', 'r') as file:
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
print("ours error：",ours.mean(),"ours std：",ours.std())
all_data.append(ours)

with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买3/csv/apperance.txt', 'r') as file:
    content = file.read()
apperance = []
for i in content.split("\n"):
    if i =='':
        continue
    if i[0]=='/' :
        continue
    if float(i.split()[-4])>shangjie:
        apperance.append(shangjie)
        continue
    if float(i.split()[-4])<xiajie:
        apperance.append(xiajie)
        continue
    apperance.append(float(i.split()[-4]))
apperance = np.array(apperance)
print("apperance error：",apperance.mean(),"apperance std：",apperance.std())
all_data.append(apperance)


with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买3/csv/2DGS.txt', 'r') as file:
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
print("GS2D error：",GS2D.mean(),"GS2D std：",GS2D.std())
all_data.append(GS2D)

with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/购买3/csv/3DGS_lightinstance.txt', 'r') as file:
    content = file.read()
GS_lightinstance = []
for i in content.split("\n"):
    if i =='':
        continue
    if i[0]=='/' :
        continue
    if float(i.split()[-1])>shangjie:
        GS_lightinstance.append(shangjie)
        continue
    if float(i.split()[-1])<xiajie:
        GS_lightinstance.append(xiajie)
        continue
    GS_lightinstance.append(float(i.split()[-1]))
GS_lightinstance = np.array(GS_lightinstance)
print("GS_lightinstance error：",GS_lightinstance.mean(),"GS std：",GS_lightinstance.std())
all_data.append(GS_lightinstance)

all_bad_points = []
for add in np.linspace(0, 0.3, 1000):
    bad_points_cat = []
    Q3 = GS.mean() + stm_beishu * GS.std()
    Q3 = add
    bad_points = np.sum(GS < Q3) / GS.shape[0]
    bad_points_cat.append(bad_points)
    # print("GS:",bad_points)

    Q3 = GS_lightinstance.mean() + stm_beishu * GS_lightinstance.std()
    Q3 = add
    bad_points = np.sum(GS_lightinstance < Q3) / GS_lightinstance.shape[0]
    bad_points_cat.append(bad_points)
    # print("colmap:",bad_points)

    Q3 = ours.mean() + stm_beishu * ours.std()
    Q3 = add
    bad_points = np.sum(ours < Q3) / ours.shape[0]
    bad_points_cat.append(bad_points)
    # print("ours:",bad_points)

    Q3 = apperance.mean() + stm_beishu * apperance.std()
    Q3 = add
    bad_points = np.sum(apperance < Q3) / apperance.shape[0]
    bad_points_cat.append(bad_points)
    # print("2DGS:",bad_points)
    all_bad_points.append(bad_points_cat)

all_bad_points = np.array(all_bad_points).T

# with plt.style.context(["science"]):

fig, ax = plt.subplots()
x = np.linspace(0, 1, 1000)
ax.plot(x, all_bad_points[0], label='3DGS',linewidth=3.0)
ax.plot(x, all_bad_points[1], label='3DGS+LightInstance',linewidth=3.0)
ax.plot(x, all_bad_points[3], label='3DGS+ApperanceEmbedding',linewidth=3.0)
ax.plot(x, all_bad_points[2], label='Ours',linewidth=3.0)
# ax.legend(fontsize = 15)
ax.set_xlabel('CD Error Thershold', fontsize=20)
ax.set_ylabel('Percentage', fontsize=20)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# ax.set_yscale('log')

plt.show()

print("end")















