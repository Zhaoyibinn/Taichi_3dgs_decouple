import torch
import torch.nn as nn
from dataclasses import dataclass
from pytorch_msssim import ssim
from dataclass_wizard import YAMLWizard


class LossFunction(nn.Module):
    @dataclass
    class LossFunctionConfig(YAMLWizard):
        lambda_value: float = 0.2
        enable_regularization: bool = True
        regularization_weight: float = 2


    def __init__(self, config: LossFunctionConfig):
        super().__init__()
        self.iteration = 0
        self.config = config


    def forward(self, iteration,decouple_image,predicted_image, ground_truth_image, point_invalid_mask=None, pointcloud_features=None):
        """
        L = (1 − 𝜆)L1 + 𝜆LD-SSIM
        predicted_image: (B, C, H, W) or (C, H, W)
        ground_truth_image: (B, C, H, W) or (C, H, W)
        """
        self.iteration = iteration
        if len(predicted_image.shape) == 3:
            predicted_image = predicted_image.unsqueeze(0)
        if len(ground_truth_image.shape) == 3:
            ground_truth_image = ground_truth_image.unsqueeze(0)

        # predicted_image = torch.tensor(predicted_image, requires_grad=True)



        mask_loss = nn.Parameter(torch.empty(1).to("cuda"))
        # print(self.iteration)
        LD_SSIM = 1 - ssim(predicted_image, ground_truth_image,data_range=1, size_average=True)
        # L1 = torch.abs(predicted_image - ground_truth_image).mean()
        # L1 = torch.abs(predicted_image - ground_truth_image).mean()
        l1loss = torch.nn.L1Loss()
        L1= l1loss(decouple_image, ground_truth_image[0])
        lambda_l = min(max(iteration-3000/30000,0),1)*(0.2-self.config.lambda_value)+self.config.lambda_value
        L = (1 - lambda_l) * L1 + lambda_l * LD_SSIM
        # L = L1


        # if pointcloud_features is not None and self.config.enable_regularization:
        #     regularization_loss = self._regularization_loss(point_invalid_mask, pointcloud_features)
        #     L = L + self.config.regularization_weight * regularization_loss
        #
        return L, L1, LD_SSIM,mask_loss

    def _regularization_loss(self, point_invalid_mask, pointcloud_features):
        """ add regularization loss to pointcloud_features, especially for s.
        exp(s) is the length of three-major axis of the ellipsoid. we don't want
        it to be too large. first we try L2 regularization.

        Args:
            pointcloud_features (_type_): _description_
        """
        s = pointcloud_features[point_invalid_mask == 0, 4:7]
        exp_s = torch.exp(s)
        regularization_loss = torch.norm(exp_s, dim=1).mean()
        return regularization_loss
        
        

