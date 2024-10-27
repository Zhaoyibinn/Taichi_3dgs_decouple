import cv2
import os
import glob
import numpy as np

# 图片源文件夹路径
source_folder = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera/C0028_darker/imgs（复件）'
# 图片保存文件夹路径
destination_folder = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera/C0028_darker/imgs'
# 缩放比例，例如0.5表示缩小到原图的50%


# 确保保存图片的文件夹存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 获取源文件夹中所有图片的路径
image_paths = glob.glob(os.path.join(source_folder, '*'))

for image_path in image_paths:
    # 读取图片
    image = cv2.imread(image_path)
    if image is not None:
        # 计算缩放尺寸
        height, width = image.shape[:2]
        new_width = int(1920)
        new_height = int(1080)

        # 缩放图片
        # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_v = np.array(img_hsv[:, :, 2], dtype=np.int64)
        img_v = (img_v * 1.1-40)
        img_v[img_v < 0] = 0
        img_hsv[:, :, 2] = np.array(img_v, dtype=np.uint8)
        resized_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        # 构建保存缩放图片的文件名
        filename = os.path.basename(image_path)
        filename = filename.rsplit(".")[0] + ".png"

        destination_path = os.path.join(destination_folder, filename)

        # 保存缩放后的图片
        cv2.imwrite(destination_path, resized_image)
        print(f"Resized and saved {filename} to {destination_folder}")
    else:
        print(f"Could not read image {image_path}")

print("Image resizing and saving process completed.")