import matplotlib.pyplot as plt
import argparse
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer

import os

def update_config(config):
    # 这里zyb手动修改了config，只需要输入根目录，但是子文件夹名称需要一致
    data_root_dir= config.data_root_dir
    config.train_dataset_json_path = os.path.join(data_root_dir, "train.json")
    config.val_dataset_json_path = os.path.join(data_root_dir, "val.json")

    config.mask_train_dataset_json_path = os.path.join(data_root_dir, "train_mask.json")
    config.mask_val_dataset_json_path = os.path.join(data_root_dir, "val_mask.json")

    config.pointcloud_parquet_path = os.path.join(data_root_dir, "point_cloud.parquet")
    return config

if __name__ == "__main__":
    plt.switch_backend("agg")
    parser = argparse.ArgumentParser("Train a Gaussian Point Cloud Scene")
    parser.add_argument("--train_config", default="config/own_config.yaml",type=str)
    parser.add_argument("--gen_template_only",
                        action="store_true", default=False)
    parser.add_argument("--vis",default=False)
    # rerun可视化
    # parser.add_argument("--decouple", default=True)
    # 是否需要加入外观解耦的模型
    args = parser.parse_args()
    if args.gen_template_only:
        config = GaussianPointCloudTrainer.TrainConfig()
        # convert config to yaml
        config.to_yaml_file(args.train_config)
        exit(0)
    config = GaussianPointCloudTrainer.TrainConfig.from_yaml_file(
        args.train_config)
    
    config = update_config(config)
    trainer = GaussianPointCloudTrainer(config)

    trainer.train(args)
