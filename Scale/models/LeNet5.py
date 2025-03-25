import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import models.Attention as Att
# Deformable ConvNets v2: More Deformable, Better Results
import models.deform_conv_v2 as DC
import models.LPS_core.LogPoolingCovDis as LPS
import models.retinal.modules as RetinalM

from models.RAMLPM.modules import (
    retina_polar,
)
import models.retinal.retinalNet as RN
import models.Foveated_convolutions.main as FoveatedConv
from torchvision.transforms import ToPILImage

show = ToPILImage()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class LeNet5(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        num_classes: 分类的数量

        """
        super(LeNet5, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # print(x.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(x.cpu().numpy()[0].reshape(112, 112))
        # plt.colorbar()
        # plt.show()
        x1 = F.relu(self.conv1(x))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, x2, x3, x4, x5]

# 0.125对应放大8, 2对应缩小0.5, 最终是[0.5-8]
scales1 = np.array([0.125, 0.149, 0.177, 0.21, 0.25, 0.297, 0.354,
                    0.42, 0.5, 0.595, 0.707, 0.841, 1, 1.189, 1.141, 1.682, 2])
scales2 = np.array([0.125, 0.149, 0.177, 0.21, 0.25, 0.297, 0.354,
                    0.42, 0.5, 0.595, 0.707, 0.841, 1, 1.189, 1.141, 1.682, 2])
scales_1 = np.array([0.125, 0.149, 0.177, 0.21, 0.25, 0.297, 0.354, 0.42,
                     0.5, 0.595, 0.707, 0.841, 1, 1.189, 1.141, 1.682, 2, 2.5, 3, 3.5, 4])
scales11 = np.array([0.25, 0.297, 0.354,
                     0.42, 0.5, 0.595, 0.707, 0.841, 1, 1.189, 1.141, 1.682, 2])

## LeNet5_Retinal_learnw_free_max1 is TICNN
class LeNet5_Retinal_learnw_free_max1(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=10,
                 retinal_H=112,
                 retinal_W=112,
                 image_H=112,
                 image_W=112, w_scale=1):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Retinal_learnw_free_max1, self).__init__()

        
        self.num_classes = num_classes
        self.retinal_teacher = RN.Retinal_learnw_teacher(
            r_min=0.01,
            r_max=1.2,
            # r_min=0.05,
            # r_max=0.8,
            image_H=image_H,  # 指的是恢复图像的大小
            image_W=image_W,
            retinal_H=retinal_H,
            retinal_W=retinal_W,
            upsampling_factor_r=1,
            upsampling_factor_theta=1,
            log_r=True,
            channel=1,
            r=0.5,
            w_scale=1,  # consistently equal to 1
            w_rotation=np.pi * 2,
        )
        self.retinal_org = RN.Retinal_learnw_org(
            r_min=0.01,
            r_max=1.2,
            # r_min=0.05,
            # r_max=0.8,
            image_H=image_H,  # 指的是恢复图像的大小
            image_W=image_W,
            retinal_H=retinal_H,
            retinal_W=retinal_W,
            upsampling_factor_r=1,
            upsampling_factor_theta=1,
            log_r=True,
            channel=1,
            r=0.5,
            w_scale=1,  # consistently equal to 1
            w_rotation=np.pi * 2,
        )
        self.retinal = RN.Retinal_1_scale2_large11(
            r_min=0.01,
            r_max=1.2,
            image_H=image_H,
            image_W=image_W,
            retinal_H=retinal_H,
            retinal_W=retinal_W,
            upsampling_factor_r=1,
            upsampling_factor_theta=1,
            log_r=True,
            channel=1,
            r=0.5,
            w_scale=1,  # consistently equal to 1
            w_rotation=np.pi * 2,
        )
        self.w_scale = w_scale #对应文中的 t_r #对应文中的 t_r
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x, test=True, scale_compensate=1):
        batchsize_, *_ = x.shape
        l_t = torch.zeros(batchsize_, 2).cuda()
        # l_t = torch.Tensor(x.size()[0], 2).uniform_(0, 0).cuda()  # 这个其实是相当于中心点, 在这个log polar的实现中，(0,0)就是中心点
        scale_ = None
        weight_s = None
        weight_r = None
        n_ = 1
        if not test:
            selected_number = random.choice([0, 1])
            if selected_number == 0:
                scale_ = torch.FloatTensor(batchsize_, 2).uniform_(0.125, 8).cuda()
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
            else:
                scale_ = torch.FloatTensor(batchsize_, 2).uniform_(0.125, 8).cuda()
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)

                g_t, x = self.retinal_org(x, l_t, 0, torch.zeros(batchsize_, 2).cuda())
                i_t = torch.cat([x, i_t], 0)
                n_ = 2
        else:
            g_t, x_ = self.retinal_org(x, l_t, 0, torch.zeros(batchsize_, 2).cuda())
            g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
            i_t = torch.cat([x_, i_t], 0)
            n_ = 2

        x1 = F.relu(self.conv1(i_t))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)

        x = torch.stack(x5.split([batchsize_] * n_))
        x = x.view(n_, batchsize_ * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(n_))) + \
            torch.logsumexp(x, dim=0)
        y = x.view(batchsize_, self.num_classes)

        # y = F.softmax(x5, dim=1)
        return y, [weight_s, weight_r, scale_, x1, x2, x3, x4, x5]
loss_f2 = torch.nn.MSELoss()  # 与pattern的损失函数

#以下是其他模型
class LeNet5_LPS(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Bing Su and Ji-Rong Wen, "Log-Polar Space Convolution Layers", Neurips 2022.
        """
        super(LeNet5_LPS, self).__init__()
        self.inplanes = 16
        
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes - self.inplanes // 4, kernel_size=5, stride=2, padding=2)
        self.logpl = LPS.LogPoolingCovLayer(
            8, 8, stride=4, pool_type='avg_pool', num_levels=1, ang_levels=4, facbase=2)
        self.lpsc1 = nn.Conv2d(in_channels, self.inplanes //
                               4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.centerconv1 = nn.Conv2d(
            in_channels, self.inplanes // 4, kernel_size=1, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # import time
        # print(x.shape)
        x1 = self.conv1(x)
        x_ = self.logpl(x)
        # print(x1.shape,x_.shape,self.lpsc1(x_).shape, self.centerconv1(x).shape)
        x2 = self.lpsc1(x_) + self.centerconv1(x)
        x1 = torch.cat((x1, x2), 1)
        x1 = F.relu(self.bn1(x1))
        # print(att1.shape)
        # print('1',time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, x2, x3, x4, x5]

class LeNet5_LPT(nn.Module):
    '''
    log polar transformation
    Human eye inspired log-polar pre-processing for neural networks  Leendert A Remmelzwaal  2020
    这个地方用的是RAMPLM的代码，用了同心圆，我们自己的话不要用
    '''

    def __init__(self, in_channels, num_classes,
                 r_min=0.05,
                 r_max=0.8,
                 H=112,
                 W=112,
                 upsampling_factor_r=1,
                 upsampling_factor_theta=1,
                 log_r=True,
                 kernel_sizes_conv2d=[[3, 3], [3, 3], [3, 3]],
                 kernel_sizes_pool=[[1, 1], [1, 1], [3, 3]],
                 strides_pool=[[1, 1], [1, 1], [3, 3]],
                 kernel_dims=[1, 32, 64, 64],
                 ):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_LPT, self).__init__()
        kernel_dims[0] = in_channels
        self.retina = retina_polar(
            r_min,
            r_max,
            H,
            W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
        )
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 这个其实是相当于中心点, 在这个log polar的实现中，(0,0)就是中心点
        l_t = torch.Tensor(x.size()[0], 2).uniform_(0, 0).cuda()
        # print(l_t)
        g_t = self.retina(x, l_t)

        # import matplotlib.pyplot as plt
        # plt.imshow(x.cpu().numpy()[0].reshape(112, 112))
        # plt.colorbar()
        # plt.show()
        # plt.close()
        # plt.imshow(g_t.cpu().numpy()[0].reshape(112, 112))
        # plt.colorbar()
        # plt.show()
        # plt.close()

        # print(g_t.shape)
        x1 = F.relu(self.conv1(g_t))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [g_t, x1, x2, x3, x4, x5]

class LeNet5_sc(nn.Module):
    # scale channels
    def __init__(self, in_channels=1, num_classes=10,
                 ):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_sc, self).__init__()
        
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def scale_channel(self, x, scales):
        b, c, h, w = x.shape
        x_scales = []
        for scale in scales:
            if scale < 1:  # 实现大scale
                # 中心裁剪然后放大
                h_ = int(h * scale)
                w_ = int(w * scale)
                top_crop = (h - h_) // 2
                right_crop = (w - w_) // 2
                scaled_image = x[:, :, top_crop:h_ +
                                                top_crop, right_crop:w_ + right_crop]

                scaled_image = F.interpolate(scaled_image, size=(
                    h, w), mode='bilinear', align_corners=False)
            elif scale > 1:  # 实现小scale
                # 填充周围后再resize成原来的
                h_ = int(h * scale)
                w_ = int(w * scale)
                pad_height = max(h - h_, 0)
                pad_width = max(w - w_, 0)

                top_pad = pad_height // 2
                bottom_pad = pad_height - top_pad
                left_pad = pad_width // 2
                right_pad = pad_width - left_pad
                scaled_image = F.pad(
                    x, (left_pad, right_pad, top_pad, bottom_pad), mode='replicate')
                scaled_image = F.interpolate(scaled_image, size=(
                    h, w), mode='bilinear', align_corners=False)
            else:
                scaled_image = x
            # import matplotlib.pyplot as plt
            # plt.imshow(scaled_image.cpu().detach().numpy()[0][0])
            # plt.colorbar()
            # plt.show()
            # # plt.savefig(str(i+3)+'.jpg')
            # plt.close()
            x_scales.append(scaled_image)

        return torch.cat(x_scales, 0)

    def forward(self, x, test=True):
        batch_size, c, h, w = x.shape
        i_t = self.scale_channel(x, scales2)
        x1 = F.relu(self.conv1(i_t))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        # format according to number of samples
        x = torch.stack(x5.split([batch_size] * len(scales2)))
        x = x.view(len(scales2), batch_size * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(len(scales2)))) + \
            torch.logsumexp(x, dim=0)
        y = x.view(batch_size, self.num_classes)
        return y, [scales2, x1, x2, x3, x4, x5]

class LeNet5_Dilation(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        第一层改为空洞卷积
        """
        super(LeNet5_Dilation, self).__init__()

        
        self.conv1 = nn.Conv2d(
            in_channels, 6, kernel_size=5, dilation=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, x2, x3, x4, x5]

class LeNet5_DC(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_DC, self).__init__()
        
        self.conv1 = DC.DeformConv2d(in_channels, 6, kernel_size=5)
        # self.att1 = DC.DeformConv2d(6,6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        # att1 = self.att1(x1)
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, x2, x3, x4, x5]

class LeNet5_Att_FoveatedC(nn.Module):
    # 只包含Foveated Conv
    def __init__(self, in_channels, num_classes, input_size=112):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_FoveatedC, self).__init__()
        
        self.conv1 = FoveatedConv.FoveatedConv2d(
            in_channels, 16, input_size=input_size)
        # self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        # self.att1 = Att.STN(in_channels=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # att1 = self.att1(x) # STN感觉更适合在第一层
        x1 = F.relu(self.conv1(x))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, x2, x3, x4, x5]

class LeNet5_Att_FoveatedC_STN(nn.Module):
    # 包含Foveated Conv 和 STN
    def __init__(self, in_channels, num_classes, input_size=112):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_FoveatedC_STN, self).__init__()
        
        self.conv1 = FoveatedConv.FoveaNet(
            in_channels, 16, num_classes, True, pool=True, input_size=input_size)
        # self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        # self.att1 = Att.STN(in_channels=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        # self.avgpool = nn.AdaptiveAvgPool2d((5,5))
        # self.line1 = nn.Linear(16*5*5, 120) 
        # self.line2 = nn.Linear(120, 84)
        # self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # att1 = self.att1(x) # STN感觉更适合在第一层
        x1 = self.conv1(x)
        return x1, [x1]

class LeNet5_Att_MSFoveatedC_STN(nn.Module):
    # 对每个Foveated 做STN
    def __init__(self, in_channels, num_classes, input_size=112):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_MSFoveatedC_STN, self).__init__()
        
        self.conv1 = FoveatedConv.MultiScaleNet(
            in_channels, 16, num_classes, True, pool=True, input_size=input_size)
        # self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        # self.att1 = Att.STN(in_channels=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        # self.avgpool = nn.AdaptiveAvgPool2d((5,5))
        # self.line1 = nn.Linear(16*5*5, 120) 
        # self.line2 = nn.Linear(120, 84)
        # self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # att1 = self.att1(x) # STN感觉更适合在第一层
        x1 = self.conv1(x)
        return x1, [x1]

class LeNet5_Att_SE(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_SE, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.att1 = Att.SEblock(6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        att1 = self.att1(x1)
        x = self.maxpool(att1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, att1, x2, x3, x4, x5]

class LeNet5_Att_CBAM(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_CBAM, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.att1 = Att.CBAM(6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        att1 = self.att1(x1)
        x = self.maxpool(att1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, att1, x2, x3, x4, x5]

class LeNet5_Att_ASPP(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.att1 = Att.ASPP(in_channels=6, atrous_rates=[
            2, 4, 8], out_channels=6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        att1 = self.att1(x1)
        x = self.maxpool(att1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, att1, x2, x3, x4, x5]

class LeNet5_Att_STN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_STN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.att1 = Att.STN(in_channels=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        att1 = self.att1(x)  # STN感觉更适合在第一层
        x1 = F.relu(self.conv1(att1))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y, [x1, att1, x2, x3, x4, x5]

class LeNet5_Att_PSTN(nn.Module):
    def __init__(self, in_channels, num_classes, samples=5):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_Att_PSTN, self).__init__()
        
        self.S = samples
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.att1 = Att.PSTN(in_channels=1, samples=samples)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x, test=True):
        batch_size, c, h, w = x.shape
        att1, theta_samples, theta_mu, beta = self.att1(
            x, None)  # STN感觉更适合在第一层

        x1 = F.relu(self.conv1(att1))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x = self.line3(x4)

        # format according to number of samples
        x = torch.stack(x.split([batch_size] * self.S))
        x = x.view(self.S, batch_size * self.num_classes)

        if not test:
            x = x.mean(dim=0)
            y = x.view(batch_size, self.num_classes)
            # y = F.softmax(y, dim=1)

        else:
            x = torch.log(torch.tensor(1.0 / float(self.S))) + \
                torch.logsumexp(x, dim=0)
            y = x.view(batch_size, self.num_classes)
            # y = F.softmax(x, dim=1)

        return y, theta_samples, beta

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, alpha):
        ctx.alpha = alpha
        ctx.save_for_backward(inp)
        return torch.where(inp < 0., alpha * inp, inp)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        ones_like_inp = torch.ones_like(inp)
        return torch.where(inp < 0., torch.zeros_like(inp),
                           ones_like_inp * ctx.alpha), None
act_fun = ActFun.apply
class LeNet5_NR(nn.Module):
    def __init__(self, in_channels, num_classes, lr=0.01, setbound=None):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_NR, self).__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.setbound = setbound
        
        if in_channels == 1:
            self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=(
                5, 5), padding=2)  # mnist填充成32
        else:
            self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=(5, 5))
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

        self.alpha1 = nn.Parameter(torch.Tensor(torch.ones(6 * 28 * 28)))
        self.alpha1.requires_grad = False
        self.alpha2 = nn.Parameter(torch.Tensor(torch.ones(16 * 10 * 10)))
        self.alpha2.requires_grad = False
        self.alpha3 = nn.Parameter(torch.Tensor(torch.ones(120)))
        self.alpha3.requires_grad = False
        self.alpha4 = nn.Parameter(torch.Tensor(torch.ones(84)))
        self.alpha4.requires_grad = False
        # self.alpha6 = nn.Parameter(torch.Tensor(torch.ones(10)))

    def update(self, x, alpha):
        # 对alpha的更新与x有关
        code = x.clone()
        code[code > 0] = 1
        code = F.softmax(code.flatten(1).sum(0))  # 用的越多，被增强的概率越大
        if self.setbound != None:
            code = alpha + code * self.lr
            code[code > 2] = 2
            return code
        return alpha + code * self.lr

    def forward(self, x, train=False, all_epoch=100, now_epoch=1):
        b, c, h, w = x.shape
        x1 = self.conv1(x)
        x1 = act_fun(x1, self.alpha1.view(
            6, 28, 28).unsqueeze(0).repeat(b, 1, 1, 1))
        x = self.maxpool(x1)

        x2 = self.conv2(x)
        x2 = act_fun(x2, self.alpha2.view(
            16, 10, 10).unsqueeze(0).repeat(b, 1, 1, 1))
        x = self.avgpool(x2)

        x = torch.flatten(x, 1)
        x3 = self.line1(x)
        x3 = act_fun(x3, self.alpha3.unsqueeze(0).repeat(b, 1))

        x4 = self.line2(x3)
        x4 = act_fun(x4, self.alpha4.unsqueeze(0).repeat(b, 1))

        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)

        if train:
            # 以概率p更新，该概率与当前epoch有关
            p = random.random()
            if (1 - now_epoch / all_epoch) > p:
                self.alpha1.data = self.update(x1, self.alpha1)
                self.alpha2.data = self.update(x2, self.alpha2)
                self.alpha3.data = self.update(x3, self.alpha3)
                self.alpha4.data = self.update(x4, self.alpha4)
            # print(self.alpha4)
        return y, [x3, x4]

from models.etn.etn import coordinates, networks, transformers
class LeNet5_(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.line1 = nn.Linear(16 * 5 * 5, 120) 
        self.line2 = nn.Linear(120, 84)
        self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # print(x.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(x.cpu().numpy()[0].reshape(112, 112))
        # plt.colorbar()
        # plt.show()
        x1 = F.relu(self.conv1(x))
        x = self.maxpool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.avgpool(x2)
        x = x.flatten(1)
        x3 = F.relu(self.line1(x))
        x4 = F.relu(self.line2(x3))
        x5 = self.line3(x4)
        y = F.softmax(x5, dim=1)
        return y
class LeNet5_ETN(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        """
        num_classes: 分类的数量

        """
        super(LeNet5_ETN, self).__init__()
        tfs = [transformers.Scale]
        network = LeNet5_(in_channels, num_classes)
        tf_default_opts = {
            'in_channels': 1,
            'kernel_size': 3,
            'nf': 32,
            'strides': (2, 1),
        }
        pose_module = networks.EquivariantPosePredictor
        seq = transformers.TransformerSequence(*[tf(pose_module, **tf_default_opts) for tf in tfs])

        self.model = self._build_model(net=network, transformer=seq, coords=coordinates.identity_grid, downsample=1)

    def _build_model(self, net, transformer, coords, downsample):
        return networks.TransformerCNN(
            net=net,
            transformer=transformer,
            coords=coords,
            downsample=downsample)

    def forward(self, x):

        y = self.model(x)
        return y, [x]
    
    
