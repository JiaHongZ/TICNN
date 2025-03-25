import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import models.Attention as Att
import models.LPS_core.LogPoolingCovDis as LPS
from models.retinal.retinal_modules import *
class Retinal_1_scale2_large11(nn.Module):
    def __init__(self,
                 r_min = 0.05,
                 r_max = 0.8,
                 image_H=112,
                 image_W=112,
                 retinal_H=112,
                 retinal_W=112,
                 upsampling_factor_r=1,
                 upsampling_factor_theta=1,
                 log_r=True,
                 channel=1,
                 r=0.5,
                 w_scale=1, # 弃置不用
                 w_rotation=np.pi*2,
                 ):
        """
        num_classes: 分类的数量
        
        """
        super(Retinal_1_scale2_large11, self).__init__()
        # r_max = np.linalg.norm(np.array([56, 56]))
        # self.retina = retina_polar_scale2_large1(
        #     r_min,
        #     r_max,
        #     retinal_H,
        #     retinal_W,
        #     upsampling_factor_r,
        #     upsampling_factor_theta,
        #     log_r,
        #     channel=channel,
        #     r=r,
        #     w_scale=w_scale,
        #     w_rotation=w_rotation,
        # )
        self.retina = retina_polar_scale2_large11(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
            channel=channel,
            r=r,
            w_scale=w_scale,
            w_rotation=w_rotation,
        )
        self.inverse_retina = inverse_retina_polar_batch_fixed(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            image_H,
            image_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
        )
    def forward(self, x, l_t, w=1):
        # 位置为[-1,1]，归一化后了
        # print(l_t)
        # weight_s, weight_r, g_t = self.retina(x, l_t, w)
        weight_s, weight_r, g_t = self.retina(x, l_t, w)
        # inverse其实不用location
        i_t = self.inverse_retina(g_t, l_t)
        return g_t, i_t, weight_s, weight_r
class Retinal_1_scale2_large11_ImageNet(nn.Module):
    def __init__(self,
                 r_min = 0.05,
                 r_max = 0.8,
                 image_H=112,
                 image_W=112,
                 retinal_H=112,
                 retinal_W=112,
                 upsampling_factor_r=1,
                 upsampling_factor_theta=1,
                 log_r=True,
                 channel=1,
                 r=0.5,
                 w_scale=1, # 弃置不用
                 w_rotation=np.pi*2,
                 ):
        """
        num_classes: 分类的数量
        
        """
        super(Retinal_1_scale2_large11_ImageNet, self).__init__()
        # r_max = np.linalg.norm(np.array([56, 56]))
        # self.retina = retina_polar_scale2_large1(
        #     r_min,
        #     r_max,
        #     retinal_H,
        #     retinal_W,
        #     upsampling_factor_r,
        #     upsampling_factor_theta,
        #     log_r,
        #     channel=channel,
        #     r=r,
        #     w_scale=w_scale,
        #     w_rotation=w_rotation,
        # )
        self.retina = retina_polar_scale2_large11_ImageNet(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
            channel=channel,
            r=r,
            w_scale=w_scale,
            w_rotation=w_rotation,
        )
        self.inverse_retina = inverse_retina_polar_batch_fixed(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            image_H,
            image_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
        )
    def forward(self, x, l_t, w=1, test=False):
        # 位置为[-1,1]，归一化后了
        # print(l_t)
        # weight_s, weight_r, g_t = self.retina(x, l_t, w)
        weight_s, weight_r, g_t = self.retina(x, l_t, w, test)
        # inverse其实不用location
        i_t = self.inverse_retina(g_t, l_t)
        return g_t, i_t, weight_s, weight_r

class Retinal_learnw_teacher(nn.Module):
    '''
    '''
    def __init__(self,
                 r_min = 0.05,
                 r_max = 0.8,
                 image_H=112,
                 image_W=112,
                 retinal_H=112,
                 retinal_W=112,
                 image_H_r=112,
                 image_W_r=112,
                 upsampling_factor_r=1,
                 upsampling_factor_theta=1,
                 log_r=True,
                 channel=1,
                 r=0.5,
                 w_scale=1, # 弃置不用
                 w_rotation=np.pi*2,
                 ):
        """
        num_classes: 分类的数量
        
        """
        super(Retinal_learnw_teacher, self).__init__()
        # r_max = np.linalg.norm(np.array([56, 56]))
        self.retina = retina_polar_learnw_teacher(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
            channel=channel,
            r=r,
            w_scale=w_scale,
            w_rotation=w_rotation,
        )
        self.inverse_retina = inverse_retina_polar_batch_fixed(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            image_H,
            image_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
        )
    def forward(self, x, l_t, s_t, w=1):
        # 位置为[-1,1]，归一化后了
        # print(l_t)
        g_t = self.retina(x, l_t, s_t, w)
        i_t = self.inverse_retina(g_t, l_t)
        return g_t, i_t
class Retinal_learnw_org(nn.Module):
    '''
    '''
    def __init__(self,
                 r_min = 0.05,
                 r_max = 0.8,
                 image_H=112,
                 image_W=112,
                 retinal_H=112,
                 retinal_W=112,
                 image_H_r=112,
                 image_W_r=112,
                 upsampling_factor_r=1,
                 upsampling_factor_theta=1,
                 log_r=True,
                 channel=1,
                 r=0.5,
                 w_scale=1, # 弃置不用
                 w_rotation=np.pi*2,
                 ):
        """
        num_classes: 分类的数量
        
        """
        super(Retinal_learnw_org, self).__init__()
        # r_max = np.linalg.norm(np.array([56, 56]))
        self.retina = retina_polar_learnw_org(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
            channel=channel,
            r=r,
            w_scale=w_scale,
            w_rotation=w_rotation,
        )
        self.inverse_retina = inverse_retina_polar_batch_fixed(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            image_H,
            image_W,
            upsampling_factor_r,
            upsampling_factor_theta,
            log_r,
        )
    def forward(self, x, l_t, s_t, w=1):
        # 位置为[-1,1]，归一化后了
        # print(l_t)
        g_t = self.retina(x, l_t, s_t, w)
        i_t = self.inverse_retina(g_t, l_t)
        return g_t, i_t

