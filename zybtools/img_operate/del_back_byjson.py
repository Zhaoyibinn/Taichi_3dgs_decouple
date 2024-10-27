import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera/imgs_operate/selected_25/imgs'
mask_path = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera/imgs_operate/selected_25/masks'
out_path = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera/imgs_operate/selected_25/final'

# 使用os.listdir()获取文件夹内的所有文件和文件夹名
entries = os.listdir(mask_path)
# 过滤出文件名，排除文件夹
files_mask = [file for file in entries if os.path.isfile(os.path.join(mask_path, file))]
files_mask.sort(key=lambda x:int(x[6:-9]))

entries = os.listdir(img_path)
# 过滤出文件名，排除文件夹
files_img = [file for file in entries if os.path.isfile(os.path.join(img_path, file))]
files_img.sort(key=lambda x:int(x[6:-4]))

if len(files_mask) != len(files_img):
    print("mask和img个数不一样")
    exit()

for i in range(len(files_mask)):
    img = cv2.imread(img_path + "/" + files_img[i])
    mask = cv2.imread(mask_path + "/" + files_mask[i])
    masked_img = img * (mask/255)
    cv2.imwrite(out_path + "/" + files_img[i],masked_img)
    # print("end")


print("end")