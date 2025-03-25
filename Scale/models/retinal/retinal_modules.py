import math
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchex.nn as exnn

# used in MNIST and Caltech101
class retina_polar_scale2_large11(nn.Module):
    """
    
    
    """
    def __init__(
        self,
        r_min=0.05,
        r_max=0.8,
        H=5,
        W=12,
        upsampling_factor_r=10, 
        upsampling_factor_theta=10, 
        log_r=True,
        channel=1,
        r=0.5,
        w_scale=1,
        w_rotation=np.pi * 2,
    ):
        super(retina_polar_scale2_large11, self).__init__()
        if log_r:
            sample_r_log = np.linspace(
                np.log(r_min), np.log(r_max), num=upsampling_factor_r * H,
            )
            sample_r = np.exp(sample_r_log)

        else:
            sample_r = np.linspace(r_min, r_max, num=upsampling_factor_r * H)

        grid_2d = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 2] 
        )
        angles = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 1]
        )
        for h in range(H * upsampling_factor_r):
            radius = sample_r[h]
            for w in range(W * upsampling_factor_theta):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    
                    [radius, radius]
                )
                angles[h, w] = torch.Tensor(
                    [angle]
                )
        self.w_scale = w_scale
        self.w_rotation = w_rotation
        self.H = H
        self.W = W
        self.register_buffer("radius", grid_2d)
        self.register_buffer("angles", angles)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

        self.global_avg_pool_s = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.AdaptiveAvgPool2d(3)
        )
        self.global_avg_pool_r = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.AdaptiveAvgPool2d(3)
        )
        # 全连接层
        self.fc_s = nn.Sequential(
            nn.Linear(32*3*3, 64),  # int(channel * r)取整数
            # nn.BatchNorm1d(64),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.fc_r = nn.Sequential(
            nn.Linear(32*3*3, 64),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        
        self.fc_s[2].weight.data.zero_()
        # self.fc_s[2].bias.data.copy_(torch.tensor([-2.7725], dtype=torch.float))
        self.fc_s[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        self.fc_r[2].weight.data.zero_()
        self.fc_r[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        # self.fc_r[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
    # self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])
    def get_grid(self, weight_s, weight_r):
        b,_ = weight_s.shape
        radius = self.radius[None].clone().repeat(b,1,1,1)
        angles = self.angles[None].clone().repeat(b,1,1,1)
        radius = radius * weight_s[:,0].view(b, 1, 1, 1) * self.w_scale  # b, 112, 112, 2
        angle = angles - weight_r[:,0].view(b, 1, 1, 1) * self.w_rotation # b, 112, 112, 1

        grid = torch.zeros_like(radius).cuda()
        grid[:,:,:,0] = radius[:,:,:,0] * torch.sin(angle[:,:,:,0])
        grid[:,:,:,1] = radius[:,:,:,1] * torch.cos(angle[:,:,:,0])
        return grid

    def att(self, x):
        branch_s = self.global_avg_pool_s(x)
        branch_s = branch_s.view(branch_s.size(0), -1)
        weight_s = self.fc_s(branch_s)
        branch_r = self.global_avg_pool_r(x)
        branch_r = branch_r.view(branch_r.size(0), -1)
        weight_r = self.fc_r(branch_r)
        return weight_s, weight_r

    def forward(self, x, l_t_prev, w=1):
        # 这里l_t_prev无用
        weight_s, weight_r = self.att(x)
        weight_s = weight_s * w
        # print(weight_s[0], weight_r[0])
        # weight = torch.ones_like(weight)
        # print(l_t_prev.view(-1, 1, 1, 2))
        grid_2d_batch = self.get_grid(weight_s, weight_r) + l_t_prev.view(-1, 1, 1, 2)
        sampled_points = F.grid_sample(x, grid_2d_batch)
        # print(sampled_points.shape)
        # print(sampled_points)
        sampled_points = self.avg_pool(sampled_points)
        return weight_s, weight_r, sampled_points

class retina_polar_scale2_large11_ImageNet(nn.Module):
    """
    
    
    """
    def __init__(
        self,
        r_min=0.05,
        r_max=0.8,
        H=5,
        W=12,
        upsampling_factor_r=10, 
        upsampling_factor_theta=10, 
        log_r=True,
        channel=1,
        r=0.5,
        w_scale=1,
        w_rotation=np.pi * 2,
    ):
        super(retina_polar_scale2_large11_ImageNet, self).__init__()
        if log_r:
            sample_r_log = np.linspace(
                np.log(r_min), np.log(r_max), num=upsampling_factor_r * H,
            )
            sample_r = np.exp(sample_r_log)

        else:
            sample_r = np.linspace(r_min, r_max, num=upsampling_factor_r * H)

        grid_2d = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 2]
        )
        angles = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 1]
        )
        for h in range(H * upsampling_factor_r):
            radius = sample_r[h]
            for w in range(W * upsampling_factor_theta):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    
                    [radius, radius]
                )
                angles[h, w] = torch.Tensor(
                    [angle]
                )
        self.w_scale = w_scale
        self.w_rotation = w_rotation
        self.H = H
        self.W = W
        self.register_buffer("radius", grid_2d)
        self.register_buffer("angles", angles)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

        self.global_avg_pool_s = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3)
        )
        self.global_avg_pool_r = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3)
        )
        # 全连接层
        self.fc_s = nn.Sequential(
            nn.Linear(64*3*3, 128),  # int(channel * r)取整数
            # nn.BatchNorm1d(64),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc_r = nn.Sequential(
            nn.Linear(32*3*3, 64),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        
        self.fc_s[2].weight.data.zero_()
        # self.fc_s[2].bias.data.copy_(torch.tensor([-2.7725], dtype=torch.float))
        self.fc_s[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        self.fc_r[2].weight.data.zero_()
        self.fc_r[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
        # self.fc_r[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))
    # self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])
    def get_grid(self, weight_s, weight_r):
        b,_ = weight_s.shape
        radius = self.radius[None].clone().repeat(b,1,1,1)
        angles = self.angles[None].clone().repeat(b,1,1,1)
        radius = radius * weight_s[:,0].view(b, 1, 1, 1) * self.w_scale  # b, 112, 112, 2
        angle = angles - weight_r[:,0].view(b, 1, 1, 1) * self.w_rotation # b, 112, 112, 1

        grid = torch.zeros_like(radius).cuda()
        grid[:,:,:,0] = radius[:,:,:,0] * torch.sin(angle[:,:,:,0])
        grid[:,:,:,1] = radius[:,:,:,1] * torch.cos(angle[:,:,:,0])
        return grid

    def att(self, x):
        branch_s = self.global_avg_pool_s(x)
        branch_s = branch_s.view(branch_s.size(0), -1)
        weight_s = self.fc_s(branch_s)
        branch_r = self.global_avg_pool_r(x)
        branch_r = branch_r.view(branch_r.size(0), -1)
        weight_r = self.fc_r(branch_r)
        return weight_s, weight_r

    def forward(self, x, l_t_prev, w=1, test=False):
        # 这里l_t_prev无用
        weight_s, weight_r = self.att(x)
        weight_s = weight_s * w
        weights_used = weight_s.clone()
        weights_used[weights_used>1] = 1
        # print(weight_s[0], weight_r[0])
        # weight = torch.ones_like(weight)
        # print(l_t_prev.view(-1, 1, 1, 2))
        grid_2d_batch = self.get_grid(weights_used, weight_r) + l_t_prev.view(-1, 1, 1, 2)
        sampled_points = F.grid_sample(x, grid_2d_batch)
        # print(sampled_points.shape)
        # print(sampled_points)
        sampled_points = self.avg_pool(sampled_points)
        return weight_s, weight_r, sampled_points

class retina_polar_learnw_teacher(nn.Module):
    """
    """
    def __init__(
        self,
        r_min=0.05,
        r_max=0.8,
        H=5,
        W=12,
        upsampling_factor_r=10, 
        upsampling_factor_theta=10, 
        log_r=True,
        channel=1,
        r=0.5,
        w_scale=1,
        w_rotation=np.pi * 2,
    ):
        super(retina_polar_learnw_teacher, self).__init__()
        if log_r:
            sample_r_log = np.linspace(
                np.log(r_min), np.log(r_max), num=upsampling_factor_r * H,
            )
            sample_r = np.exp(sample_r_log)

        else:
            sample_r = np.linspace(r_min, r_max, num=upsampling_factor_r * H)

        grid_2d = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 2]
        )
        angles = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 1]
        )
        for h in range(H * upsampling_factor_r):
            radius = sample_r[h]
            for w in range(W * upsampling_factor_theta):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    
                    [radius, radius]
                )
                angles[h, w] = torch.Tensor(
                    [angle]
                )
        self.w_scale = w_scale
        self.w_rotation = w_rotation
        self.H = H
        self.W = W
        self.register_buffer("radius", grid_2d)
        self.register_buffer("angles", angles)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

    def get_grid(self, b, weight):
        radius = self.radius[None].clone().repeat(b,1,1,1)
        angles = self.angles[None].clone().repeat(b,1,1,1)
        radius = radius * weight[:,0].view(b, 1, 1, 1) * self.w_scale # b, 112, 112, 2
        angle = angles - weight[:,1].view(b, 1, 1, 1) * self.w_rotation # b, 112, 112, 1

        grid = torch.zeros_like(radius).cuda()
        grid[:,:,:,0] = radius[:,:,:,0] * torch.sin(angle[:,:,:,0])
        grid[:,:,:,1] = radius[:,:,:,1] * torch.cos(angle[:,:,:,0])
        return grid


    def forward(self, x, l_t_prev, s_t, w=1):
        # 这里l_t_prev无用
        # weight_s, weight_r = self.att(x)
        # weight_s = weight_s * w
        # print(weight_s[0], weight_r[0])
        # weight = torch.ones_like(weight)
        # print(l_t_prev.view(-1, 1, 1, 2))
        batch_size, *_ = x.shape
        grid_2d_batch = self.get_grid(batch_size, w) + l_t_prev.view(-1, 1, 1, 2)
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border')
        # print(sampled_points.shape)
        # print(sampled_points)
        sampled_points = self.avg_pool(sampled_points)
        return sampled_points

class retina_polar_learnw_org(nn.Module):
    """
        不变
    """
    def __init__(
        self,
        r_min=0.05,
        r_max=0.8,
        H=5,
        W=12,
        upsampling_factor_r=10, 
        upsampling_factor_theta=10, 
        log_r=True,
        channel=1,
        r=0.5,
        w_scale=1,
        w_rotation=np.pi * 2,
    ):
        super(retina_polar_learnw_org, self).__init__()
        if log_r:
            sample_r_log = np.linspace(
                np.log(r_min), np.log(r_max), num=upsampling_factor_r * H,
            )
            sample_r = np.exp(sample_r_log)

        else:
            sample_r = np.linspace(r_min, r_max, num=upsampling_factor_r * H)

        grid_2d = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 2]
        )
        angles = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 1]
        )
        for h in range(H * upsampling_factor_r):
            radius = sample_r[h]
            for w in range(W * upsampling_factor_theta):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    
                    [radius, radius]
                )
                angles[h, w] = torch.Tensor(
                    [angle]
                )
        self.w_scale = w_scale
        self.w_rotation = w_rotation
        self.H = H
        self.W = W
        self.register_buffer("radius", grid_2d)
        self.register_buffer("angles", angles)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

    def get_grid(self, b):
        radius = self.radius[None].clone().repeat(b,1,1,1)
        angles = self.angles[None].clone().repeat(b,1,1,1)
        angle = angles

        grid = torch.zeros_like(radius).cuda()
        grid[:,:,:,0] = radius[:,:,:,0] * torch.sin(angle[:,:,:,0])
        grid[:,:,:,1] = radius[:,:,:,1] * torch.cos(angle[:,:,:,0])
        return grid


    def forward(self, x, l_t_prev, s_t, w=1):
        # 这里l_t_prev无用
        # weight_s, weight_r = self.att(x)
        # weight_s = weight_s * w
        # print(weight_s[0], weight_r[0])
        # weight = torch.ones_like(weight)
        # print(l_t_prev.view(-1, 1, 1, 2))
        batch_size, *_ = x.shape
        grid_2d_batch = self.get_grid(batch_size) + l_t_prev.view(-1, 1, 1, 2)
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border')
        # print(sampled_points.shape)
        # print(sampled_points)
        sampled_points = self.avg_pool(sampled_points)
        return sampled_points

class inverse_retina_polar_batch_fixed(nn.Module):
    def __init__(
        self,
        r_min=0.01,
        r_max=0.6,
        retinal_H=5,
        retinal_W=5,
        H=5,
        W=12,
        upsampling_factor_r=10, 
        upsampling_factor_theta=10, 
        log_r=True,
    ):
        super(inverse_retina_polar_batch_fixed, self).__init__()
        self.H = H
        self.W = W
        self.r_min = r_min
        self.r_max = r_max
        grid_2d = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 2]
        )
        for i in range(H):
            for j in range(W):
                # 归一化到单位圆上的点
                x = (i - int(H/2))/(H/2) # 这里除以H/2的依据是，r轴长H/2就覆盖了整个面积，然后归一化
                y = (j - int(W/2))/(W/2)
                r = retinal_H * (np.log(np.clip(np.sqrt(x**2+y**2),1e-6,(H**2+W**2)))-np.log(r_min))/(np.log(r_max)-np.log(r_min))
                a = np.arctan2(y, x)
                a = a if a > 0 else 2.0 * np.pi + a
                t = 0.5 * a * retinal_W / np.pi
                grid_2d[i, j] = torch.Tensor(
                    [t/(retinal_W/2)-1,r/(retinal_H/2)-1]
                )
        self.register_buffer("grid_2d", grid_2d)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

    def forward(self, x, l_t_prev):
        # 这里不用location
        grid_2d_batch = l_t_prev.view(-1, 1, 1, 2) * 0 + self.grid_2d[None]
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border')
        sampled_points = self.avg_pool(sampled_points)
        return sampled_points
