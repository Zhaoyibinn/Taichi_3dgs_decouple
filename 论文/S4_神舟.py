import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  plotnine import *
import seaborn as sns
import open3d as o3d


all_data = []
shangjie = 1
xiajie = 0.001
# 使用pandas读取CSV文件
with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/3DGS.txt', 'r') as file:
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


with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/colmap.txt', 'r') as file:
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



with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/scene_instance_less.txt', 'r') as file:
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

with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/apperance.txt', 'r') as file:
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


with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/2DGS.txt', 'r') as file:
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


with open('/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/result/S4_神舟/csv/3DGS_lightinstance.txt', 'r') as file:
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



"""
编写时间：2022年2月06日 09：50

作者: 宁海涛，代码运行出错或者因包版本更新出错，请关注微信公众号【DataCharm】进行实时获取更新。

代码中Proplot包绘制的部分可能与Matplotlib绘制不同，需注意两者绘图语法的不同~~

"""
# 注意：本案例中引入的SciencePlots库主题方式会随着SciencePlots版本的更新发生改变，读者还需参考其官网引入的正确方式。
#      建议按照官网引入方式进行操作和使用最新版本的ScienePlots。如引入方式发生改变，请关注微信公众号【DataCharm】进行实时获取代码更新。

"""
编写时间：2023年8月19日 15：20（修正版）

作者: 宁海涛，代码运行出错或者因包版本更新出错，请关注微信公众号【DataCharm】进行实时获取更新。

代码中Proplot包绘制的部分可能与Matplotlib绘制不同，需注意两者绘图语法的不同~~

"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True




# #-----------------------------a）利用Matplotlib 绘制的同一坐标系中的多个密度图---------------
#
# fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
# plt.xscale('log')
# # plt.yscale('log')
# # for i, index,color in zip(range(len(palette)),data_df.columns,palette):
# data = colmap
# density = stats.kde.gaussian_kde(data)
# x = np.linspace(0.001,shangjie,500)
# y = density(x)
# y[0]=y[-1]=0
# ax.plot(x,y, lw=2,color="#000000")
# ax.fill(x,y,color="#65BE86",label="Colmap",alpha=.3)
#
# data = GS
# density = stats.kde.gaussian_kde(data)
# x = np.linspace(0.001,shangjie,500)
# y = density(x)
# y[0]=y[-1]=0.5
# ax.plot(x,y, lw=2,color="#000000")
# ax.fill(x,y,color="#352A87",label="3DGS",alpha=.3)
#
# data = ours
# density = stats.kde.gaussian_kde(data)
# x = np.linspace(0.001,shangjie,500)
# y = density(x)
# y[0]=y[-1]=0
# ax.plot(x,y, lw=3,color="#000000",zorder=100)
# ax.fill(x,y,color="#FFC337",label="Ours",alpha=0.5,zorder=100)
#
#
#
#
# ax.set_xlabel("Values")
# ax.set_ylabel("Density")
# ax.legend()
# # fig.savefig('\\单变量图表绘制\\图3-2-14多组密度图绘制示例_a.pdf',bbox_inches='tight')
# # fig.savefig('\\单变量图表绘制\\图3-2-14多组密度图绘制示例_a.png',
# #             bbox_inches='tight',dpi=300)
# plt.show()



# import matplotlib.pyplot as plt
#
# plt.boxplot([ours, GS,colmap], labels=['ours', 'GS','Colmap'])
# plt.ylabel('Values')
# plt.title('Boxplot of Group A and Group B')
# plt.show()




"""
编写时间：2022年4月12日 10：30

作者: 宁海涛，代码运行出错或者因包版本更新出错，请关注微信公众号【DataCharm】进行实时获取更新。
"""

"""
此代码为使用Python-Prolot包进行学术风格图表绘制，绘图语法和Matplotlib有所不同，需注意。
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as p

# data_np = np.array([ours,colmap,GS])

data_pd_3DGS = pd.DataFrame(GS,columns=['3DGS'])
data_pd_2DGS = pd.DataFrame(GS2D,columns=['2DGS'])
data_pd_colmap = pd.DataFrame(colmap,columns=['Colmap'])
data_pd_ours = pd.DataFrame(ours,columns=['Ours'])
combined_df = pd.concat([data_pd_3DGS,data_pd_2DGS,data_pd_colmap,data_pd_ours], axis=1)

colors = ["#2FBE8F","#459DFF","#FF5B9B","#FFCC37"]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = .8
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

tips = sns.load_dataset("tips")


# a）使用seaborn绘制分组箱线图
fig,ax = plt.subplots(figsize=(4,3.5),dpi=200,facecolor="w")
plt.yscale('log')
plt.grid()

# grouped_boxplot = sns.boxplot(x="day", y="total_bill", hue="smoker",
#                               data=tips, palette=colors,saturation=1,
#                               width=.7,linewidth=1.2)

# grouped_boxplot = sns.boxplot(data=combined_df, palette=colors,saturation=1,
#                               width=.7,linewidth=1.2,fliersize=3,whis=1.5,showfliers=False)

sns.violinplot(data=combined_df,palette=colors,linewidth=1,saturation=1,inner=None,ax=ax)

from numpy import mean
sns.pointplot(data=combined_df,estimator=mean,ci="sd",err_kws={'linewidth': 1.5},linestyles="",color='k',ax=ax)
ax.set_ylabel("C2C Error")


plt.tight_layout()
plt.show()

# # b）使用SciencePlots主题库绘制分组箱线图
#
# plt.style.use('science')  # 需安装SciencePlots  读者需注意其最新的引用方式
#
# fig,ax = plt.subplots(figsize=(4,3.5),dpi=100,facecolor="w")
# grouped_boxplot = sns.boxplot(x="day", y="total_bill", hue="smoker",
#                               data=tips, palette=colors,saturation=1,
#                               width=.7,linewidth=1.2)
# ax.set_xlabel("Time")
# ax.set_ylabel("Values")
# fig.savefig('\第4章 双变量图形的绘制\图4-1-24 分组箱线图绘制示例_b.pdf',bbox_inches='tight')
# fig.savefig('\第4章 双变量图形的绘制\图4-1-24 分组箱线图绘制示例_b.png',
#             bbox_inches='tight',dpi=300)
# plt.show()














