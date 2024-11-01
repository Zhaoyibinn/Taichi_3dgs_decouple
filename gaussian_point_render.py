#!/bin/python3

import argparse
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch, quaternion_to_rotation_matrix_torch
from dataclasses import dataclass
from taichi_3d_gaussian_splatting.ImagePoseDataset import ImagePoseDataset

import torch
import torchvision
from torch import nn
import numpy as np
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt

from taichi_3d_gaussian_splatting.apperance_network import AppearanceNetwork
from taichi_3d_gaussian_splatting.apperance_network import decouple_appearance

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


class GaussianPointRenderer:
    @dataclass
    class GaussianPointRendererConfig:
        parquet_path: str
        cameras: torch.Tensor
        device: str = "cuda"
        image_height: int = 544
        image_width: int = 976
        camera_intrinsics: torch.Tensor = torch.tensor(
            [[581.743, 0.0, 488.0], [0.0, 581.743, 272.0], [0.0, 0.0, 1.0]],
            device="cuda")

        def set_portrait_mode(self):
            self.image_height = 976
            self.image_width = 544
            self.camera_intrinsics = torch.tensor(
                [[1163.486, 0.0, 272.0], [0.0, 1163.486, 488.0], [0.0, 0.0, 1.0]],
                device="cuda")

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]

    def __init__(self, config: GaussianPointRendererConfig) -> None:
        self.config = config
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16
        scene = GaussianPointCloudScene.from_parquet(
            config.parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
        self.scene = self._merge_scenes([scene])
        self.scene = self.scene.to(self.config.device)
        self.cameras = self.config.cameras.to(self.config.device)
        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(
                self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_id=0,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.8,
                far_plane=1000.,
                depth_to_sort_key_scale=100.))

        self.appearance_network = AppearanceNetwork(64 + 3, 3).to("cuda")
        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(100, 64).to("cuda"))
        self._appearance_embeddings.data.normal_(0, std)


    def _merge_scenes(self, scene_list):
        # the config does not matter here, only for training

        merged_point_cloud = torch.cat(
            [scene.point_cloud for scene in scene_list], dim=0)
        merged_point_cloud_features = torch.cat(
            [scene.point_cloud_features for scene in scene_list], dim=0)
        num_of_points_list = [scene.point_cloud.shape[0]
                              for scene in scene_list]
        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
        end_offset_list = np.cumsum(num_of_points_list).tolist()
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                center=scene_list[idx].point_cloud.mean(dim=0),
                visible=True
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        point_object_id = torch.zeros(
            (merged_point_cloud.shape[0],), dtype=torch.int32, device=self.config.device)
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            point_object_id=point_object_id,
            config=GaussianPointCloudScene.PointCloudSceneConfig(
                max_num_points_ratio=None
            ))
        return merged_scene

    def run(self, output_prefix,args):
        decoupled_model_path = args.decoupled_model_path
        decoupled_vector_path = args.decoupled_vector_path
        decouple = args.decouple
        if decouple:
            # self.appearance_network.load_state_dict(torch.load(decoupled_model_path))
            self.appearance_network = torch.load(decoupled_model_path)
            self._appearance_embeddings = torch.load(decoupled_vector_path, map_location=torch.device('cuda'))
        num_cameras = self.cameras.shape[0]
        for idx, val_data in enumerate(tqdm(val_data_loader)):
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            image_gt, q, t, camera_info, i = val_data
        # for i in tqdm(range(num_cameras)):
            # c = self.cameras[i, :, :].unsqueeze(0)
            # q, t = SE3_to_quaternion_and_translation_torch(c)
            # image_gt, q, t, camera_info,index = next(
            #     val_data_loader_iter)
            with torch.no_grad():
                image, depth, pixel_valid_point_count = self.rasteriser(
                    GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        point_object_id=self.scene.point_object_id,
                        camera_info=self.camera_info,
                        q_pointcloud_camera=q,
                        t_pointcloud_camera=t,
                        color_max_sh_band=3,
                    )
                )

                # depth_points = np.indices((800, 800)).reshape(2, -1)
                # depth_points = np.append(depth_points,depth_points[0]).reshape([3,640000])
                # depth_points[2] =np.array(depth.cpu().detach()).flatten() * 100

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(depth_points.T)
                # o3d.visualization.draw_geometries([pcd])
                image = image.permute(2, 0, 1)

                if decouple:
                    appearance_embedding = self.get_apperance_embedding(i)
                    decouple_image, transformation_map = decouple_appearance(image, appearance_embedding,
                                                                             self.appearance_network,q,t)

                    image = decouple_image.permute(1, 2, 0)

                img = Image.fromarray(torch.clamp(image * 255, 0, 255).byte().cpu().numpy(), 'RGB')

                img.save(output_prefix / f'frame_{i:03}.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str,default="logs/tat_train_experiment_downsample_warmup/scene_29000.parquet")
    parser.add_argument("--decoupled_model_path", type=str,
                        default="logs/tat_train_experiment_downsample_warmup/model_29000.pth")
    parser.add_argument("--decoupled_vector_path", type=str,
                        default="logs/tat_train_experiment_downsample_warmup/apperance_vector_29000.pth")
    parser.add_argument("--poses", type=str, default="data/S1/train.json",help="could be a .pt file that was saved as torch.save(), or a json dataset file used by Taichi-GS")
    parser.add_argument("--output_prefix", type=str,default="test_render")
    parser.add_argument("--decouple", default=True)


    parser.add_argument("--gt_prefix", type=str, default="")
    parser.add_argument("--portrait_mode", action='store_true', default=False)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)

    output_prefix = Path(args.output_prefix)
    os.makedirs(output_prefix, exist_ok=True)
    if args.gt_prefix:
        gt_prefix = Path(args.gt_prefix)
        os.makedirs(gt_prefix, exist_ok=True)
    else:
        gt_prefix = None

    if args.poses.endswith(".pt"):
        config = GaussianPointRenderer.GaussianPointRendererConfig(
            args.parquet_path, torch.load(args.poses))
        if args.portrait_mode:
            config.set_portrait_mode()
    elif args.poses.endswith(".json"):
        val_dataset = ImagePoseDataset(
            dataset_json_path=args.poses)
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=0)
        val_data_loader_iter = cycle(val_data_loader)

        cameras = torch.zeros((len(val_data_loader), 4, 4))
        camera_info = None
        for idx, val_data in enumerate(tqdm(val_data_loader)):
            try:
                image_gt, q, t, camera_info = val_data
            except:
                # print("存在语义")
                image_gt, q, t, camera_info,_ = val_data
            r = quaternion_to_rotation_matrix_torch(q)
            cameras[idx, :3, :3] = r
            cameras[idx, :3, 3] = t
            cameras[idx, 3, 3] = 1.0
            # dump autoscaled GT images at the resolution of training
            if gt_prefix is not None:
                img = torchvision.transforms.functional.to_pil_image(image_gt)
                img.save(gt_prefix / f'frame_{idx:03}.png')
        config = GaussianPointRenderer.GaussianPointRendererConfig(
            args.parquet_path, cameras
        )
        # override camera meta data as provided
        config.image_width = camera_info.camera_width
        config.image_height = camera_info.camera_height
        config.camera_intrinsics = camera_info.camera_intrinsics
    else:
        raise ValueError(f"Unrecognized poses file format: {args.poses}, Must be .pt or .json file")

    renderer = GaussianPointRenderer(config)
    renderer.run(output_prefix,args)
