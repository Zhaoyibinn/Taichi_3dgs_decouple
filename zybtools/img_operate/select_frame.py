import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 设置你想要读取的文件夹路径
folder_path = '/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera_2/2/imgs_operate/C0034'
save_folder_path = '/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera_2/2/imgs_operate/selected'


# 使用os.listdir()获取文件夹内的所有文件和文件夹名
entries = os.listdir(folder_path)
# 过滤出文件名，排除文件夹
files = [file for file in entries if os.path.isfile(os.path.join(folder_path, file))]
files.sort(key=lambda x:int(x[6:-4]))



# 打印所有文件名
for i in range(files.__len__()):
    # if i >=1022:
    #     break
    if i % 7 == 0:
        print(files[i])
        img = cv2.imread(folder_path+"/"+files[i])
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img_v = np.array(img_hsv[:, :, 2], dtype=np.int64)
        img_v = (img_v -0)
        img_v[img_v<0]=0
        img_hsv[:, :, 2] = np.array(img_v, dtype=np.uint8)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(save_folder_path+"/"+files[i],img)
