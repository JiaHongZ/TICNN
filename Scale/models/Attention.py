import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import distributions

'''
non-local
2018
'''
# --------------------------------
# no-local block
# Non-Local Recurrent Network for Image Restoration
#---------------------------------
class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

'''
DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
https://arxiv.org/abs/1606.00915
Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
2017
'''
# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

'''
Squeeze-and-Excitation Networks
http://xxx.itp.ac.cn/abs/1709.01507
Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
CVPR 2018 paper, accepted by TPAMI
'''
class SEblock(nn.Module):
    def __init__(self, channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(SEblock, self).__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )
    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        # 全连接层得到权重
        weight = self.fc(branch)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        scale = weight * x
        return scale

'''---------------------------------------------------'''
'''
CBAM: Convolutional Block Attention Module
Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
2018
'''
class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x


'''---------------------------------------------------'''

'''
STN
Spatial Transformer Networks
2015 Jaderberg
'''
class STN(nn.Module):
    def __init__(self, in_channels):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3) # 新增为了能适应尺度---可以改动
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # print(x.size(), theta.size())
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x
class STN_ScaleChannels(nn.Module):
    def __init__(self, in_channels):
        super(STN_ScaleChannels, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3) # 新增为了能适应尺度---可以改动
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        mode = 'bilinear' if self.training else 'nearest'
        x1 = F.grid_sample(x, grid, mode=mode)
        x2 = F.grid_sample(F.interpolate(x, scale_factor=2), grid, mode=mode)
        x3 = F.grid_sample(F.interpolate(x, scale_factor=4), grid, mode=mode)
        tmp = torch.cat((x1, x2, x3), dim=1)
        tmp.detach = lambda: x1.detach()  # HACK!
        return tmp

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x

# visualization stn
# def convert_image_np(inp):
#     """Convert a Tensor to numpy image."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     return inp
# def visualize_stn():
#     with torch.no_grad():
#         # Get a batch of training data
#         data = next(iter(test_loader))[0].to(device)
#
#         input_tensor = data.cpu()
#         transformed_input_tensor = model.stn(data).cpu()
#
#         in_grid = convert_image_np(
#             torchvision.utils.make_grid(input_tensor))
#
#         out_grid = convert_image_np(
#             torchvision.utils.make_grid(transformed_input_tensor))
#
#         # Plot the results side-by-side
#         f, axarr = plt.subplots(1, 2)
#         axarr[0].imshow(in_grid)
#         axarr[0].set_title('Dataset Images')
#
#         axarr[1].imshow(out_grid)
#         axarr[1].set_title('Transformed Images')
#
# # Visualize the STN transformation on some input batch
# visualize_stn()
#
# plt.ioff()
# plt.show()

'''
Schwöbel, Pola, et al. "Probabilistic spatial transformer networks." Uncertainty in Artificial Intelligence. 2022.
'''



class PSTN(nn.Module):
    def __init__(self, in_channels, samples=5):
        super(PSTN, self).__init__()
        # Spatial transformer localization-network
        self.S = samples
        self.N = in_channels
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3) # 新增为了能适应尺度---可以改动
        )
        # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(10 * 3 * 3, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 3 * 2)
        # )
        self.alpha_p = 1
        self.theta_dim = 3*2
        self.fc_loc_mu = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, self.theta_dim)
        )
        self.fc_loc_beta = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, self.theta_dim),
            # add activation function for positivity
            nn.Softplus())  # beta needs to be positive, and also small so maybe a logscale parametrisation would be better

        # Initialize the weights/bias with identity transformation
        self.fc_loc_mu[2].weight.data.zero_()
        self.fc_loc_mu[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc_beta[2].weight.data.zero_()
        self.fc_loc_beta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def make_affine_matrix(self, theta, scale_x, scale_y, translation_x, translation_y):
        # theta is rotation angle in radians
        a = scale_x * torch.cos(theta)
        b = - torch.sin(theta)
        c = translation_x

        d = torch.sin(theta)
        e = scale_y * torch.cos(theta)
        f = translation_y

        param_tensor = torch.stack([a, b, c, d, e, f], dim=-1)

        affine_matrix = param_tensor.view([-1, 2, 3])
        return affine_matrix
    def make_affine_parameters(self, params):
        if params.shape[-1] == 1:  # only learn rotation
            angle = params[:, 0]
            scale = torch.ones([params.shape[0]], device=params.device)
            translation_x = torch.zeros([params.shape[0]], device=params.device)
            translation_y = torch.zeros([params.shape[0]], device=params.device)
            affine_matrix = self.make_affine_matrix(angle, scale, scale, translation_x, translation_y)

        if params.shape[-1] == 2:  # only perform crop - fix scale and rotation.
            theta = torch.zeros([params.shape[0]], device=params.device)
            scale_x = 0.5 * torch.ones([params.shape[0]], device=params.device)
            scale_y = 0.5 * torch.ones([params.shape[0]], device=params.device)
            translation_x = params[:, 0]
            translation_y = params[:, 1]
            affine_matrix = self.make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

        elif params.shape[-1] == 3:  # crop with learned scale, isotropic, and tx/tx
            theta = torch.zeros([params.shape[0]], device=params.device)
            scale_x = params[:, 0]
            scale_y = params[:, 0]
            translation_x = params[:, 1]
            translation_y = params[:, 2]
            affine_matrix = self.make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

        elif params.shape[-1] == 4:  # "full afffine" with isotropic scale
            theta = params[:, 0]
            scale = params[:, 1]
            scale_x, scale_y = scale, scale
            translation_x = params[:, 2]
            translation_y = params[:, 3]
            affine_matrix = self.make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

        elif params.shape[-1] == 5:  # "full afffine" with anisotropic scale
            theta = params[:, 0]
            scale_x = params[:, 1]
            scale_y = params[:, 2]
            translation_x = params[:, 3]
            translation_y = params[:, 4]
            affine_matrix = self.make_affine_matrix(theta, scale_x, scale_y, translation_x, translation_y)

        elif params.shape[-1] == 6:  # full affine, raw parameters
            affine_matrix = params.view([-1, 2, 3])

        return affine_matrix  # [S * bs, 2, 3]
    def forward_localizer(self, x, x_high_res):
        if x_high_res is None:
            x_high_res = x
        batch_size, c, h, w = x_high_res.shape
        _, _, small_h, small_w = x.shape

        theta_mu, beta = self.compute_theta_beta(x)

        # repeat x in the batch dim so we avoid for loop
        # (this doesn't do anything for N=1)
        x_high_res = x_high_res.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(self.N * batch_size, c, h, w)
        theta_mu_upsample = theta_mu.view(batch_size * self.N,
                                          self.theta_dim)  # mean is the same for all S: [bs * N, theta_dim]
        beta_upsample = beta.view(batch_size * self.N,
                                  self.theta_dim)  # variance is also the same, difference comes in through sampling
        alpha_upsample = self.alpha_p * torch.ones_like(theta_mu_upsample)  # upsample scalar alpha

        # make the T-dist object and sample it here?
        # it's apparently ok to generate distribution anew in each forward pass (e.g. https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # maybe we could do this more efficiently because of the independence assumptions within theta?
        T_dist = distributions.studentT.StudentT(df=2 * alpha_upsample, loc=theta_mu_upsample,
                                                 scale=torch.sqrt(beta_upsample / alpha_upsample))
        theta_samples = T_dist.rsample([self.S])  # shape: [self.S, batch_size, self.theta_dim]
        theta_samples = theta_samples.view([self.S * batch_size, self.theta_dim])

        # repeat for the number of samples
        x_high_res = x_high_res.repeat(self.S, 1, 1, 1)
        x_high_res = x_high_res.view([self.S * batch_size, c, h, w])
        affine_params = self.make_affine_parameters(theta_samples)
        big_grid = F.affine_grid(affine_params, x_high_res.size())
        small_grid = F.interpolate(big_grid.permute(0, 3, 1, 2), size=(small_h, small_w), mode="nearest").permute(0, 2, 3, 1)
        x = F.grid_sample(x_high_res, small_grid)
        # theta samples: [S, bs, nr_params]
        return x, theta_samples, (theta_mu, beta)
    def compute_theta_beta(self, x):
        batch_size = x.shape[0]
        x = self.localization(x)
        x = x.view(batch_size, -1)

        theta_mu = self.fc_loc_mu(x)
        beta = self.fc_loc_beta(x)
        return theta_mu, beta
    def forward(self, x, x_high_res):
        # get input shape
        batch_size = x.shape[0]

        # get output for pstn module
        x, theta_samples, (theta_mu, beta) = self.forward_localizer(x, x_high_res) # fix alpha for now
        return x, theta_samples, theta_mu, beta

