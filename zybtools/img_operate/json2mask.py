def json():
    import os
    import json
    from PIL import Image, ImageDraw

    def convert_to_tuple(points):
        return [tuple(p) for p in points]

    # 设置输入输出文件夹路径和文件格式
    json_folder = "/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera/imgs_operate/selected_25/json"  # json文件放的那个目录
    mask_folder = "/media/zhaoyibin/common/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/data/real_camera/imgs_operate/selected_25/masks"  # 转化成mask保存到哪个目录
    img_format = "png"

    # 遍历json文件夹中的所有json文件
    for json_file in os.listdir(json_folder):
        if not json_file.endswith(".json"):  # 看看取出来的图片格式是不是json格式的文件，如果不是的话，直接取下一个文件
            continue
        # 读取json文件中的信息
        # os.path.join(json_folder, json_file)得到每一个json文件的绝对地址，接着按照这个地址，json.load(f) 打开这个json文件
        with open(os.path.join(json_folder, json_file), "r") as f:
            data = json.load(f)

        # 获取原图尺寸和标注信息
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]
        shapes = data["shapes"]

        # 创建空白的二值图像,全黑的彩色图像
        mask = Image.new("RGB", (img_width, img_height), 0)  # 和标注图同等大小的全黑图片，彩色模式
        draw = ImageDraw.Draw(mask)  # 使用 PIL（Python Imaging Library）库的 ImageDraw 模块创建一个可绘制的对象 draw
        # 绘制标注信息
        for shape in shapes:  # 依次遍历一个json文件保存的多个标注信息
            points = shape["points"]  # 取出某一个标注信息的坐标
            points = convert_to_tuple(points)  # points 转换为一个元组的列表。元组不可改变其值
            label = shape["label"]  # 取出该标注的标签信息
            shape_type = shape["shape_type"]
            if shape_type == "rectangle":
                if label == "body":  # 确保取出的 标注的标签是 line
                    draw.rectangle(points, fill=(218,218,218))
                    # draw.rectangle(points, fill=(255,0,0))

                elif label == "fanban":  # # 确保取出的 标注的标签是 car
                    draw.rectangle(points, fill=(36, 36, 36))
                    # draw.rectangle(points, fill=(0,255,0))
            if shape_type == "polygon":

                if label == "sat":  # 确保取出的 标注的标签是 line
                    draw.polygon(points, fill=(255, 255, 255))
                    # draw.polygon(points, fill=(255,0,0))

                elif label == "_background_":  # # 确保取出的 标注的标签是 car
                    draw.polygon(points, fill=(0, 0, 0))
                # elif label == "jieshouqi":  # # 确保取出的 标注的标签是 car
                #     draw.polygon(points, fill=(112, 112, 112))
                    # draw.polygon(points, fill=(0,255,0))

        # 保存标注图像
        mask_file = os.path.splitext(json_file)[0] +"_mask"+ "." + img_format
        mask.save(os.path.join(mask_folder, mask_file))


if __name__ == '__main__':
    json()
