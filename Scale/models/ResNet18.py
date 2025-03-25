import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torchvision import models
import models.Attention as Att
from torchvision.transforms import ToPILImage
import models.Foveated_convolutions.main as FoveatedConv
show = ToPILImage()
import models.deform_conv_v2 as DC  # Deformable ConvNets v2: More Deformable, Better Results
import models.LPS_core.LogPoolingCovDis as LPS
import models.RAM_core.modules as RAMModules
import models.retinal.modules as RetinalM

from models.RAMLPM.modules import (
    retina_polar,
)
from models.retinal.modules import (
    retina_polar2,
    inverse_retina_polar_batch_fixed,
    inverse_retina_polar_batch_att,
)
import models.retinal.retinalNet as RN

# scales_mnist = np.array([0.4, 0.5, 0.25, 2, 2.5, 4]) # 模拟人类的远近
scales_caltech = np.array([0.25, 0.5, 2, 3])


# scales3 = np.array([0.125, 0.25, 0.5, 2, 4, 8])

def rescale_image(image, h, w, scales):
    rescaled_images = []

    for scale in scales:
        # Rescale the image using bilinear interpolation
        scaled_image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)

        # Calculate the dimensions of the rescaled image
        height = scaled_image.shape[2]
        width = scaled_image.shape[3]

        if height < h or width < w:
            # Apply border padding to reach (112, 112) size
            pad_height = max(h - height, 0)
            pad_width = max(w - width, 0)

            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad

            scaled_image = F.pad(scaled_image, (left_pad, right_pad, top_pad, bottom_pad))
        elif height > h or width > w:
            # Apply cropping to reach (112, 112) size
            # crop_height = min(height, 112)
            # crop_width = min(width, 112)

            top_crop = (height - h) // 2
            right_crop = (width - w) // 2

            scaled_image = scaled_image[:, :, top_crop:h + top_crop, right_crop:w + right_crop]

        rescaled_images.append(scaled_image)

    return rescaled_images


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


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


scales1 = np.array(
    [0.125, 0.149, 0.177, 0.21, 0.25, 0.297, 0.354, 0.42, 0.5, 0.595, 0.707, 0.841, 1, 1.189, 1.141, 1.682, 2])
scales2 = np.array(
    [0.25, 0.297, 0.354, 0.42, 0.5, 0.595, 0.707, 0.841, 1, 1.189, 1.141, 1.682, 2, 2.381, 2.825, 3.367, 4])
scales_imagenet = np.array(
    [0.25, 1])
scales_imagenet1 = np.array(
    [1, 4])
import random



# Caltech101
class ResNet18_Retinal_learnw_free1(nn.Module):
    def __init__(self, in_channels=3, num_classes=100,
                 retinal_H=224,
                 retinal_W=224,
                 image_H=224,
                 image_W=224,
                 w_scale=4):
        super(ResNet18_Retinal_learnw_free1, self).__init__()
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
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
            w_scale=1,  # 这个权重不需要了
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x, test=True, scale_compensate=[1, 1]):
        batch_size, *_ = x.shape

        l_t = torch.zeros(batch_size, 2).cuda()
        scale_ = None
        weight_s = None
        weight_r = None
        n_ = 1
        # print('scale_compensate',scale_compensate)
        if not test:
            selected_number = random.choice([0, 1])
            if selected_number == 0:
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
            else:
                # scale_ = torch.ones(batch_size, 2).cuda() / scale_compensate
                # scale_[:, 1] = scale_[:, 1] * 0
                # g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)

                g_t, x = self.retinal_org(x, l_t, 0, torch.zeros(batch_size, 2).cuda())
                i_t = torch.cat([x, i_t], 0)
                n_ = 2
        else:
            g_t, x_ = self.retinal_org(x, l_t, 0, torch.zeros(batch_size, 2).cuda())
            g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
            # g_t, x = self.retinal_teacher(x, l_t, 0, torch.ones(batch_size, 2).cuda())
            i_t = torch.cat([x_, i_t], 0)
            n_ = 2

        # x_scaled_lp.append(i_t)
        # print(weight_s)
        # import matplotlib.pyplot as plt
        # plt.imshow(show(x[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('0.jpg')
        # plt.close()
        # # plt.imshow(g_t.detach().cpu().numpy()[0].reshape(112, 112))
        # # plt.colorbar()
        # # # plt.show()
        # # plt.savefig('1.jpg')
        # # plt.close()
        # plt.imshow(show(i_t[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('2.jpg')
        # plt.close()
        x = self.conv1(i_t)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        x = torch.stack(x.split([batch_size] * n_))
        x = x.view(n_, batch_size * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(n_))) + \
            torch.logsumexp(x, dim=0)
        y = x.view(batch_size, self.num_classes)
        # print(weight_s[0])
        return y, [weight_s, weight_r, scale_, x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
class ResNet18_Retinal_learnw_free1_max1(nn.Module):
    def __init__(self, in_channels=3, num_classes=100,
                 retinal_H=224,
                 retinal_W=224,
                 image_H=224,
                 image_W=224,
                 w_scale=1):
        super(ResNet18_Retinal_learnw_free1_max1, self).__init__()
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
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
            w_scale=1,  # 这个权重不需要了
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x, test=True, scale_compensate=[1, 1]):
        batch_size, *_ = x.shape

        l_t = torch.zeros(batch_size, 2).cuda()
        scale_ = None
        weight_s = None
        weight_r = None
        n_ = 1
        # print('scale_compensate',scale_compensate)
        if not test:
            selected_number = random.choice([0, 1])
            if selected_number == 0:
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
            else:
                # scale_ = torch.ones(batch_size, 2).cuda() / scale_compensate
                # scale_[:, 1] = scale_[:, 1] * 0
                # g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)

                g_t, x = self.retinal_org(x, l_t, 0, torch.zeros(batch_size, 2).cuda())
                i_t = torch.cat([x, i_t], 0)
                n_ = 2
        else:
            g_t, x_ = self.retinal_org(x, l_t, 0, torch.zeros(batch_size, 2).cuda())
            g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
            # g_t, x = self.retinal_teacher(x, l_t, 0, torch.ones(batch_size, 2).cuda())
            i_t = torch.cat([x_, i_t], 0)
            n_ = 2

        # x_scaled_lp.append(i_t)
        # print(weight_s)
        # import matplotlib.pyplot as plt
        # plt.imshow(show(x[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('0.jpg')
        # plt.close()
        # # plt.imshow(g_t.detach().cpu().numpy()[0].reshape(112, 112))
        # # plt.colorbar()
        # # # plt.show()
        # # plt.savefig('1.jpg')
        # # plt.close()
        # plt.imshow(show(i_t[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('2.jpg')
        # plt.close()
        x = self.conv1(i_t)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        x = torch.stack(x.split([batch_size] * n_))
        x = x.view(n_, batch_size * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(n_))) + \
            torch.logsumexp(x, dim=0)
        y = x.view(batch_size, self.num_classes)
        # print(weight_s[0])
        return y, [weight_s, weight_r, scale_, x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
class ResNet18_Retinal_learnw_free1_max1_UseImageNetRetinal(nn.Module):
    def __init__(self, in_channels=3, num_classes=100,
                 retinal_H=224,
                 retinal_W=224,
                 image_H=224,
                 image_W=224,
                 w_scale=1):
        super(ResNet18_Retinal_learnw_free1_max1_UseImageNetRetinal, self).__init__()
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
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
            w_scale=1,  # 这个权重不需要了
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )

        self.w_scale = w_scale
        self.num_classes = num_classes
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x, retinalmodel, test=True, scale_compensate=[1, 1]):

            
        batch_size, *_ = x.shape

        l_t = torch.zeros(batch_size, 2).cuda()
        scale_ = None
        weight_s = None
        weight_r = None
        # print('scale_compensate',scale_compensate)
        g_t, x_ = self.retinal_org(x, l_t, 0, torch.zeros(batch_size, 2).cuda())
        g_t, i_t, weight_s, weight_r = retinalmodel(x, l_t, self.w_scale)
        # g_t, x = self.retinal_teacher(x, l_t, 0, torch.ones(batch_size, 2).cuda())
        i_t = torch.cat([x_, i_t], 0)
        n_ = 2

        # x_scaled_lp.append(i_t)
        # print(weight_s)
        # import matplotlib.pyplot as plt
        # plt.imshow(show(x[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('0.jpg')
        # plt.close()
        # # plt.imshow(g_t.detach().cpu().numpy()[0].reshape(112, 112))
        # # plt.colorbar()
        # # # plt.show()
        # # plt.savefig('1.jpg')
        # # plt.close()
        # plt.imshow(show(i_t[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('2.jpg')
        # plt.close()
        x = self.conv1(i_t)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        x = torch.stack(x.split([batch_size] * n_))
        x = x.view(n_, batch_size * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(n_))) + \
            torch.logsumexp(x, dim=0)
        y = x.view(batch_size, self.num_classes)
        # print(weight_s[0])
        return y, [weight_s, weight_r, scale_, x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_Retinal_learnw_free1_max1_Retinal(nn.Module):
    def __init__(self, in_channels=3, num_classes=100,
                 retinal_H=224,
                 retinal_W=224,
                 image_H=224,
                 image_W=224,
                 w_scale=1):
        super(ResNet18_Retinal_learnw_free1_max1_Retinal, self).__init__()
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
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
            w_scale=1,  # 这个权重不需要了
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        self.w_scale = w_scale
    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x, test=True, scale_compensate=[1, 1]):
        batch_size, *_ = x.shape

        l_t = torch.zeros(batch_size, 2).cuda()
        scale_ = None
        weight_s = None
        weight_r = None
        n_ = 1
        # print('scale_compensate',scale_compensate)
        if not test:
            selected_number = random.choice([0, 1])
            if selected_number == 0:
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
            else:
                # scale_ = torch.ones(batch_size, 2).cuda() / scale_compensate
                # scale_[:, 1] = scale_[:, 1] * 0
                # g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
        else:
            g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)

        return x, [weight_s, weight_r, scale_]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
class ResNet18_Retinal_learnw_free1_ImageNet_Pretrain(nn.Module):
    def __init__(self, in_channels=3, num_classes=100,
                 retinal_H=224,
                 retinal_W=224,
                 image_H=224,
                 image_W=224,
                 w_scale=4):
        super(ResNet18_Retinal_learnw_free1_ImageNet_Pretrain, self).__init__()
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
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
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        self.retinal = RN.Retinal_1_scale2_large11_ImageNet(
            r_min=0.01,
            r_max=1.2,
            image_H=image_H,
            image_W=image_W,
            retinal_H=retinal_H,
            retinal_W=retinal_W,
            upsampling_factor_r=1,
            upsampling_factor_theta=1,
            log_r=True,
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.in_channels = 64
        self.model = models.resnet18(pretrained=True)


    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x, test=True, scale_compensate=[1, 1], img_org=None):
        batch_size, *_ = x.shape

        l_t = torch.zeros(batch_size, 2).cuda()
        scale_ = None
        weight_s = None
        weight_r = None
        n_ = 1
        # print('scale_compensate',scale_compensate)
        if not test:
            selected_number = random.choice([0,1])
            if selected_number == 0:
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.125, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
            else:
                # scale_ = torch.ones(batch_size, 2).cuda() / scale_compensate
                # scale_[:, 1] = scale_[:, 1] * 0
                # g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.125, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)

                g_t, x = self.retinal_org(img_org, l_t, 0, torch.zeros(batch_size, 2).cuda())
                i_t = torch.cat([x, i_t], 0)
                n_ = 2
        else:
            g_t, x_ = self.retinal_org(x, l_t, 0, torch.zeros(batch_size, 2).cuda())
            g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
            # g_t, x = self.retinal_teacher(x, l_t, 0, torch.ones(batch_size, 2).cuda())
            i_t = torch.cat([x_, i_t], 0)
            n_ = 2
        # x_scaled_lp.append(i_t)
        # print(weight_s)
        # import matplotlib.pyplot as plt
        # plt.imshow(show(x[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('0.jpg')
        # plt.close()
        # # plt.imshow(g_t.detach().cpu().numpy()[0].reshape(112, 112))
        # # plt.colorbar()
        # # # plt.show()
        # # plt.savefig('1.jpg')
        # # plt.close()
        # plt.imshow(show(i_t[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('2.jpg')
        # plt.close()
        x = self.model(i_t)


        x = torch.stack(x.split([batch_size] * n_))
        x = x.view(n_, batch_size * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(n_))) + \
            torch.logsumexp(x, dim=0)
        y = x.view(batch_size, self.num_classes)
        # print(weight_s[0])
        return y, [weight_s, weight_r, scale_]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

# Use pretrained eye model
# imagenet
# train eye model
class ResNet18_Retinal_learnw_free1_ImageNet_Pretrain_Retinal(nn.Module):
    def __init__(self, in_channels=3, num_classes=100,
                 retinal_H=224,
                 retinal_W=224,
                 image_H=224,
                 image_W=224,
                 w_scale=1): # 非max=1的时候改成4就好，并且注意，如果要训练4的retinal，需要在retinal_module中将[weight_s>1]=1那两行去掉
        super(ResNet18_Retinal_learnw_free1_ImageNet_Pretrain_Retinal, self).__init__()
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
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
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
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        self.retinal = RN.Retinal_1_scale2_large11_ImageNet(
            r_min=0.01,
            r_max=1.2,
            image_H=image_H,
            image_W=image_W,
            retinal_H=retinal_H,
            retinal_W=retinal_W,
            upsampling_factor_r=1,
            upsampling_factor_theta=1,
            log_r=True,
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.in_channels = 64

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x, test=True, scale_compensate=[1, 1], img_org=None):
        batch_size, *_ = x.shape

        l_t = torch.zeros(batch_size, 2).cuda()
        scale_ = None
        weight_s = None
        weight_r = None
        n_ = 1
        # print('scale_compensate',scale_compensate)
        if not test:
            selected_number = random.choice([0, 1])
            if selected_number == 0:
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
            else:
                # scale_ = torch.ones(batch_size, 2).cuda() / scale_compensate
                # scale_[:, 1] = scale_[:, 1] * 0
                # g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
                scale_ = torch.FloatTensor(batch_size, 2).uniform_(0.25, 4).cuda()  # 之后再试试 0.125 4, 注意w_scale要相应改动
                scale_[:, 1] = scale_[:, 1] * 0
                g_t, i_t = self.retinal_teacher(x, l_t, 0, scale_ * scale_compensate)
                i_t = i_t.detach()
                g_t, i_t, weight_s, weight_r = self.retinal(i_t, l_t, self.w_scale)
        else:
            g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
            # g_t, x = self.retinal_teacher(x, l_t, 0, torch.ones(batch_size, 2).cuda())
        # x_scaled_lp.append(i_t)
        # print(weight_s)
        # import matplotlib.pyplot as plt
        # plt.imshow(show(x[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('0.jpg')
        # plt.close()
        # # plt.imshow(g_t.detach().cpu().numpy()[0].reshape(112, 112))
        # # plt.colorbar()
        # # # plt.show()
        # # plt.savefig('1.jpg')
        # # plt.close()
        # plt.imshow(show(i_t[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('2.jpg')
        # plt.close()

        return i_t, [weight_s, weight_r, scale_]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
class ResNet18_Retinal_frozenRetinal_ImageNet_Pretrain(nn.Module):
    def __init__(self, in_channels=3, num_classes=100,
                 retinal_H=224,
                 retinal_W=224,
                 image_H=224,
                 image_W=224,
                 w_scale=1):
        super(ResNet18_Retinal_frozenRetinal_ImageNet_Pretrain, self).__init__()

        self.retinal = RN.Retinal_1_scale2_large11_ImageNet(
            r_min=0.01,
            r_max=1.2,
            image_H=image_H,
            image_W=image_W,
            retinal_H=retinal_H,
            retinal_W=retinal_W,
            upsampling_factor_r=1,
            upsampling_factor_theta=1,
            log_r=True,
            channel=in_channels,
            r=0.5,
            w_scale=1,  # 这个权重不需要了
            w_rotation=np.pi * 2,
        )
        retinalmodel = ResNet18_Retinal_learnw_free1_ImageNet_Pretrain_Retinal(in_channels=3, num_classes=100,
                    retinal_H=224,
                    retinal_W=224,
                    image_H=224,
                    image_W=224,
                    w_scale=1)
        retinalmodel.load_state_dict(torch.load('model_zoo/retinal.pth'))
        self.retinal = retinalmodel.retinal
        for name, param in self.retinal.named_parameters():
            param.requires_grad = False
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.in_channels = 64
        self.model = models.resnet18(pretrained=True)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x, test=True, scale_compensate=[1, 1], img_org=None):
        batch_size, *_ = x.shape

        l_t = torch.zeros(batch_size, 2).cuda()
        g_t, i_t, weight_s, weight_r = self.retinal(x, l_t, self.w_scale)
        i_t = i_t.detach()
        # g_t, x = self.retinal_teacher(x, l_t, 0, torch.ones(batch_size, 2).cuda())
        i_t = torch.cat([x, i_t], 0)
        n_ = 2
        # x_scaled_lp.append(i_t)
        # print(weight_s)
        # import matplotlib.pyplot as plt
        # plt.imshow(show(x[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('0.jpg')
        # plt.close()
        # # plt.imshow(g_t.detach().cpu().numpy()[0].reshape(112, 112))
        # # plt.colorbar()
        # # # plt.show()
        # # plt.savefig('1.jpg')
        # # plt.close()
        # plt.imshow(show(i_t[0].cpu()))
        # plt.colorbar()
        # # plt.show()
        # plt.savefig('2.jpg')
        # plt.close()
        x = self.model(i_t)


        x = torch.stack(x.split([batch_size] * n_))
        x = x.view(n_, batch_size * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(n_))) + \
            torch.logsumexp(x, dim=0)
        y = x.view(batch_size, self.num_classes)
        # print(weight_s[0])
        return y, [weight_s, weight_r]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

class ResNet18_Pretrain(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, in_channels=1, num_classes=10, requires_grad=False):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x), x

class ResNet18_Pretrain_ImageNet(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, in_channels=1, num_classes=10, requires_grad=False):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.model(x), x





class ResNet18_sc(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18_sc, self).__init__()
        self.num_classes = num_classes
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

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
                scaled_image = x[:, :, top_crop:h_ + top_crop, right_crop:w_ + right_crop]

                scaled_image = F.interpolate(scaled_image, size=(h, w), mode='bilinear', align_corners=False)
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
                scaled_image = F.pad(x, (left_pad, right_pad, top_pad, bottom_pad), mode='replicate')
                scaled_image = F.interpolate(scaled_image, size=(h, w), mode='bilinear', align_corners=False)
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
        batch_size, *_ = x.shape
        i_t = self.scale_channel(x, scales1)
        x = self.conv1(i_t)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        x = torch.stack(x.split([batch_size] * len(scales1)))
        x = x.view(len(scales1), batch_size * self.num_classes)

        x = torch.log(torch.tensor(1.0 / float(len(scales1)))) + torch.logsumexp(x, dim=0)
        y = x.view(batch_size, self.num_classes)

        return y, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

class ResNet18_LPS(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18_LPS, self).__init__()
        self.in_channels = 64
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes - self.inplanes // 4, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.logpl = LPS.LogPoolingCovLayer(8, 8, stride=4, pool_type='avg_pool', num_levels=1, ang_levels=4, facbase=2)
        self.lpsc1 = nn.Conv2d(3, self.inplanes // 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.centerconv1 = nn.Conv2d(3, self.inplanes // 4, kernel_size=1, stride=2, padding=0)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        # print(self.logpl(x).shape, self.lpsc1(self.logpl(x)).shape, self.centerconv1(x).shape)
        x2 = self.lpsc1(self.logpl(x)) + self.centerconv1(x)
        x = torch.cat((x1, x2), 1)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_LPS_Pretrain(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18_LPS_Pretrain, self).__init__()
        self.in_channels = 64
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes - self.inplanes // 4, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.logpl = LPS.LogPoolingCovLayer(8, 8, stride=4, pool_type='avg_pool', num_levels=1, ang_levels=4, facbase=2)
        self.lpsc1 = nn.Conv2d(3, self.inplanes // 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.centerconv1 = nn.Conv2d(3, self.inplanes // 4, kernel_size=1, stride=2, padding=0)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 设计匹配函数，仅加载在修改后的模型中存在的预训练参数
        resnet = models.resnet18(pretrained=True)
        pretrained_dict = resnet.state_dict()
        model_dict = self.state_dict()
        matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        # 将匹配到的预训练参数加载到修改后的模型中
        model_dict.update(matched_dict)
        self.load_state_dict(model_dict)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.lpsc1(self.logpl(x)) + self.centerconv1(x)
        x = torch.cat((x1, x2), 1)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_LPT(nn.Module):
    def __init__(self, in_channels, num_classes,
                 r_min=0.05,
                 r_max=0.8,
                 H=112,
                 W=112,
                 upsampling_factor_r=1,
                 upsampling_factor_theta=1,
                 log_r=True,
                 kernel_dims=[1, 32, 64, 64], ):
        super(ResNet18_LPT, self).__init__()
        self.in_channels = 64
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

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        l_t = torch.Tensor(x.size()[0], 2).uniform_(0, 0).cuda()  # 这个其实是相当于中心点, 在这个log polar的实现中，(0,0)就是中心点
        # print(l_t)
        g_t = self.retina(x, l_t)

        x = self.conv1(g_t)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [g_t, x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_STN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18_STN, self).__init__()
        self.in_channels = 64
        self.att1 = Att.STN(in_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        att1 = self.att1(x)  # STN感觉更适合在第一层
        x = self.conv1(att1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [att1, x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_Att_SE(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18_Att_SE, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.att1 = Att.SEblock(64)
        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        # STN感觉更适合在第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.att1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_Att_CBAM(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18_Att_CBAM, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.att1 = Att.CBAM(64)
        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        # STN感觉更适合在第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.att1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_Att_ASPP(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18_Att_ASPP, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.att1 = Att.ASPP(in_channels=64, atrous_rates=[
            2, 4, 8], out_channels=64)
        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        # STN感觉更适合在第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.att1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_Att_NONLOCAL(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18_Att_NONLOCAL, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.att1 = Att.NonLocalBlock(64)
        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        # STN感觉更适合在第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.att1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_Att_Att_FoveatedC(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=224):
        super(ResNet18_Att_Att_FoveatedC, self).__init__()
        self.in_channels = 64
        self.conv1_fov = FoveatedConv.FoveatedConv2d(in_channels, 64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.att1 = Att.NonLocalBlock(64)
        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        # STN感觉更适合在第一层
        x = self.relu(self.conv1_fov(x))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.att1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet18_DC(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18_DC, self).__init__()
        self.in_channels = 64
        self.conv1 = DC.DeformConv2d(in_channels, 64,
        kernel_size = 7, stride = 2, padding = 3)
        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        # STN感觉更适合在第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4]

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # att1 = self.att1(x) # STN感觉更适合在第一层
        x1 = self.conv1(x)
        return x1, [x1]
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

class ResNet18_Att_MSFoveatedC_STN(nn.Module):
    # 对每个Foveated 做STN
    def __init__(self, in_channels, num_classes, input_size=112):
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(ResNet18_Att_MSFoveatedC_STN, self).__init__()
        # 卷积神经网络
        self.conv1 = FoveatedConv.MultiScaleNet(
            in_channels, 64, num_classes, True, pool=True, input_size=input_size)
        # self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        # self.att1 = Att.STN(in_channels=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        # self.avgpool = nn.AdaptiveAvgPool2d((5,5))
        # self.line1 = nn.Linear(16*5*5, 120)  # 这里把第三个卷积当作是全连接层了
        # self.line2 = nn.Linear(120, 84)
        # self.line3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # att1 = self.att1(x) # STN感觉更适合在第一层
        x1 = self.conv1(x)
        return x1, [x1]

from models.etn.etn import coordinates, networks, transformers
class ResNet18_ETN(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(ResNet18_ETN, self).__init__()
        tfs = [transformers.Scale]
        network =ResNet18(in_channels, num_classes)
        tf_default_opts = {
            'in_channels': 3,
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
        return y
    
    
from models.RIC.ResNet18.RIC_ResNet18 import RIC_ResNet
class ResNet18_RIC(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=112):
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(ResNet18_RIC, self).__init__()
        self.model = RIC_ResNet(BATCH_SIZE=32, num_classes=num_classes)

    def forward(self, x):
        y = self.model(x)
        return y, [x]