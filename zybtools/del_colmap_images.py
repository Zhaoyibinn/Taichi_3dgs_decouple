

# 原始文件路径
input_file_path = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_3_colmap_less/colmap（复件）/images_all.txt'
# 新文件路径
output_file_path = '/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_3_colmap_less/colmap（复件）/images.txt'

# 打开原始文件和新文件
with open(input_file_path, 'r', encoding='utf-8') as file_reader, \
     open(output_file_path, 'w', encoding='utf-8') as file_writer:
    # 逐行读取文件
    idx = 1
    image_idx = 1
    for line in file_reader:
        # 检查是否应该删除这行
        if idx >4 and (idx%4==1 or idx%4==2):
        # 写入新文件


            idx += 1
        else:
            if idx >4 and idx%4==3:
                line_sp = line.split()
                line_sp[0] = str(image_idx)
                line_sp[-1] ="r_" + str(image_idx) + ".png"
                line = " ".join(line_sp)
                line = line + '\n'
                image_idx += 1
            if idx > 4 and idx % 4 == 0:
                line = '\n'
            file_writer.write(line)
            idx += 1

print('文件处理完成，已删除空行并保存到新文件。')