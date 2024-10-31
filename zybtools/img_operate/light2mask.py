import cv2
import numpy as np
# from torchgen.api.cpp import return_type
import os

def color2mask(path,filename,save_folder_path):

    img = cv2.imread(path)
    save_path ="test_png.png"
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    cv2.imwrite(save_folder_path+filename,gray_image)
    return gray_image


folder_path = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/taichi_data/data/black_NeRF/lower_lego/imgs/'
save_folder_path = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/taichi_data/data/black_NeRF/lower_lego/light_masks/'
entries = os.listdir(folder_path)
for entry in entries:
    # 获取文件的完整路径
    full_path = os.path.join(folder_path, entry)
    filename = entry
    color2mask(full_path, filename, save_folder_path)

