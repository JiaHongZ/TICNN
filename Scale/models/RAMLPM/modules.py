import math
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchex.nn as exnn


class retina_polar(nn.Module):
    """
    A retina (glimpse sensor) that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. The sample space is the
    region bounded by two concentric circles. The image extends -1
    to 1 in 2d Euclidean space.
    Field of view encodes the information with a high resolution around
    l, and gather data from a large area.
    Args:
        r_min: the size of the radius of the inner circle.
        r_max: the size of the radius of the outer circle.

        About r_min, rmax
        In the process of converting from Cartesian coordinates to log-polar coordinates, the r_min and r_max parameters define the radial range of the log-polar transformation. These parameters determine the minimum and maximum radial distance values that will be considered in the log-polar transformation.
        In log-polar coordinates, the radial distance r is mapped to log(r) instead of the linear r as in Cartesian coordinates. The log-polar transformation effectively warps the image, emphasizing the central region while compressing the outer regions.
        Here's how r_min and r_max affect the log-polar transformation:
        r_min: This parameter represents the minimum radial distance to be included in the log-polar transformation. Any point in the Cartesian coordinate system whose radial distance from the center (i.e., the origin) is less than r_min will not be transformed and will remain at the same position in the log-polar coordinate system.
        r_max: This parameter represents the maximum radial distance to be included in the log-polar transformation. Any point in the Cartesian coordinate system whose radial distance from the center is greater than r_max will also not be transformed and will not appear in the log-polar coordinate system.
        By adjusting r_min and r_max, you can control the extent to which the log-polar transformation focuses on the central region and the level of compression applied to the outer regions of the image. A smaller r_min will emphasize the central details, while a larger r_max will include more of the image's outer regions in the log-polar representation.

        H, W: the size of the ouput tensor.
        upsampling_factor_r, upsampling_factor_theta: the sample space
        is divided into H by W regions, and the interpolated pixel value
        is integrated over the region. the sampling factors essentially determine
        the how finely the values are sampled for the trapezoidal integration.
    Returns:
        a tensor of shape (B, C, H, W). H is the radial axis, W is the angular axis.
    """

    def __init__(
        self,
        r_min=0.01,
        r_max=0.6,
        H=5,
        W=12,
        upsampling_factor_r=10, # 对r轴进行放缩
        upsampling_factor_theta=10, # 对theta轴进行放缩
        log_r=True,
    ):
        super(retina_polar, self).__init__()
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
        for h in range(H * upsampling_factor_r):
            radius = sample_r[h]
            for w in range(W * upsampling_factor_theta):
                angle = 2 * np.pi * w / (W * upsampling_factor_theta)
                grid_2d[h, w] = torch.Tensor(
                    [radius * np.cos(angle), radius * np.sin(angle)]
                )
        self.register_buffer("grid_2d", grid_2d)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

    def forward(self, x, l_t_prev):
        grid_2d_batch = l_t_prev.view(-1, 1, 1, 2) + self.grid_2d[None]
        sampled_points = F.grid_sample(x, grid_2d_batch)
        sampled_points = self.avg_pool(sampled_points)
        return sampled_points


class retina_rectangular(nn.Module):
    """
    A retina (glimpse sensor) that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. The sample space is rectangular
    grids with different spacings. The image extends -1
    to 1 in 2d Euclidean space.
    Field of view encodes the information with a high resolution around
    l, and gather data from a large area.
    Args:
        interval: spacing in the smallest grid, relative to the size of the image.
        The image size is 2 x 2.
        g: size of the square patches in the glimpses extracted
        by the retina.
        k: The number of patches.
        s: Scaling factor for succesive patches.
    """

    def __init__(self, interval, g, k=1, s=3):
        super(retina_rectangular, self).__init__()
        grid_2d = torch.empty([g, g, 2])
        sample_x = np.array(range(g)) * interval
        sample_y = np.array(range(g)) * interval
        sample_x = sample_x - (sample_x[0] + sample_x[-1]) / 2
        sample_y = sample_y - (sample_y[0] + sample_y[-1]) / 2
        for h in range(g):
            for w in range(g):
                grid_2d[h, w] = torch.Tensor([sample_x[h], sample_y[w]])
        grid_2ds = []
        for num_patch in range(k):
            grid_2ds.append(grid_2d * (s ** (num_patch - 1)))
        self.register_buffer("grid_2ds", torch.stack(grid_2ds))

    def forward(self, x, l_t_prev):
        """Extracts patches from images around specified locations.

        Args:
            x: Batched images of shape (B, C, H, W).
            l_t_prev: Batched coordinates of shape (B, 22)
        Returns:
            A 5D tensor of shape (B, k, C, g, g, C)
        """
        sampled_points_scaled = []
        for i in range(self.grid_2ds.shape[0]):
            grid_2d = self.grid_2ds[i]
            grid_2d_batch = l_t_prev.view(-1, 1, 1, 2) + grid_2d[None]
            sampled_points = F.grid_sample(x, grid_2d_batch)
            sampled_points_scaled.append(sampled_points)
            sampled_points_scaled = torch.stack(sampled_points_scaled, 1)
        return sampled_points_scaled


class CircularPad(nn.Module):
    def __init__(self, pad_top):
        super(CircularPad, self).__init__()
        self.pad_top = pad_top

    def forward(self, x):
        top_pad_left = x[:, :, : self.pad_top, : x.shape[3] // 2]
        top_pad_right = x[:, :, : self.pad_top, x.shape[3] // 2 :]
        top_pad = torch.cat([top_pad_right, top_pad_left], 3)
        x = torch.cat([top_pad, x], 2)
        return x


class CNN_in_polar_coords(nn.Module):
    """
    CNN module with padding along the angular axis.
    Args:
         kernel_sizes_conv2d: a list of kernel sizes for conv.
         strides_conv2d: a list of strides for conv.
         kernel_sizes_pool: a list of kernel sizes for max pooling.
         kernel_dims: a list of input and output dims for conv.
                     The first element is the input channel dim of
                     the input images. The size is
                     len(kernel_sizes_conv2d) + 1.
    Returns:
        3d tensor
    """

    def __init__(
        self,
        kernel_sizes_conv2d,
        kernel_sizes_pool,
        kernel_dims,
        strides_pool,
        pool_type="max",
    ):
        super(CNN_in_polar_coords, self).__init__()
        layers = []
        for layer in range(len(kernel_sizes_conv2d)):
            layers.append(
                exnn.PeriodicPad2d(pad_left=kernel_sizes_conv2d[layer][1] - 1)
            )
            layers.append(
                torch.nn.ReplicationPad2d(
                    (0, 0, 0, (kernel_sizes_conv2d[layer][0] - 1) // 2)
                )
            )
            layers.append(CircularPad(kernel_sizes_conv2d[layer][0] // 2))
            layers.append(
                nn.Conv2d(
                    kernel_dims[layer],
                    kernel_dims[layer + 1],
                    kernel_sizes_conv2d[layer],
                )
            )
            pad_size = kernel_sizes_pool[layer][1] - strides_pool[layer][1]
            layers.append(exnn.PeriodicPad2d(pad_left=pad_size))
            if pool_type == "max":
                pool = nn.MaxPool2d
            elif pool_type == "avg":
                pool = nn.AvgPool2d
            else:
                raise ValueError("pool_type should be either 'max' or 'avg'")
            if all(ks == 1 for ks in kernel_sizes_pool[layer]):
                pass
            else:
                layers.append(
                    nn.MaxPool2d(kernel_sizes_pool[layer], stride=strides_pool[layer])
                )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(kernel_dims[layer + 1], momentum=0.01))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`
    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.
    Returns
    -------- g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """

    def __init__(self, h_g, h_l):
        super(glimpse_network, self).__init__()

        # glimpse layer
        self.fc1 = exnn.Linear(h_g)

        # location layer
        self.fc2 = exnn.Linear(h_l)

        self.fc3 = exnn.Linear(h_g + h_l)
        self.fc4 = exnn.Linear(h_g + h_l)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(x))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t


class DebugLayer(nn.Module):
    """
    A module for debugging.
    """

    def forward(self, x):
        pdb.set_trace()
        return x


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.
    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.
    In other words:
        `h_t = relu( fc(h_t_prev) + fc(g_t) )`
    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.
    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """

    def __init__(self, hidden_size):
        super(core_network, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = exnn.Linear(hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.
    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.
    Returns
    -------
    - a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = self.fc(h_t)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.
    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        # compute mean
        mu = torch.clamp(self.fc(h_t), min=-1.0, max=1.0)

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise

        # bound between [-1, 1]
        # l_t = torch.tanh(l_t)
        l_t = l_t.detach()

        return mu, l_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.
    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t.detach()))
        return b_t
