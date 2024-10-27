import cv2
import os

# 视频文件路径
video_path = '/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera_2/2/imgs_operate/C0034.MP4'
# 保存帧的文件夹路径
save_folder = '/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera_2/2/imgs_operate/C0034'

# 确保保存帧的文件夹存在
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
while True:
    # 读取视频的下一帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("Reached end of video or cannot fetch a frame.")
        break

    # 构建保存帧的文件名
    if frame_count % 5 ==0:
        frame_filename = os.path.join(save_folder, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        print("saved:",frame_count)

    # 保存帧为JPEG格式


    # 增加帧计数
    frame_count += 1

# 释放视频捕获对象
cap.release()
print(f"Saved {frame_count} frames to {save_folder}")