import argparse
import pandas as pd
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer

def save_ply(pointcloud):
    print(pointcloud.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, default="/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/taichi_data/logs/NeRF/LEGO_3dgs/parquet/scene_29000.parquet")
    parser.add_argument("--ply_path", type=str, default="/home/zhaoyibin/3DRE/3DGS/taichi_3d_gaussian_splatting_complex/taichi_data/logs/NeRF/LEGO_3dgs/parquet/scene_29000.ply")
    parser.add_argument("--train_config", default="config/own_config.yaml",type=str)

    args = parser.parse_args()
    config = GaussianPointCloudTrainer.TrainConfig.from_yaml_file(
        args.train_config)
    scene = GaussianPointCloudScene.from_parquet(
        args.parquet_path, config,config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
    scene.to_ply(args.ply_path)