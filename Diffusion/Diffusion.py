import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    从给定的系数向量中提取与指定时间步长相对应的系数，并将其 reshape 成形如 
    [batch_size, 1, 1, 1, ...] 的形状，以便进行广播操作。
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

# 增加噪声
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model  # 传入的去噪模型
        self.T = T  # 扩散过程的总步数

        # betas 是扩散过程中每个时间步的噪声强度，从 beta_1 线性插值到 beta_T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas  # alphas 是扩散过程中的保留因子，表示输入数据保留的比例
        alphas_bar = torch.cumprod(alphas, dim=0)  # alphas_bar 是 alphas 的累积乘积，用于表示从第 1 步到 t 步的总保留比例

        # sqrt_alphas_bar 和 sqrt_one_minus_alphas_bar 是扩散过程中的两个关键系数，用于加噪数据
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))  # 用于乘以原始数据 x_0
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))  # 用于乘以噪声

    def forward(self, x_0):
        """
        实现扩散模型的前向传播（训练过程），根据输入 x_0 加入噪声，并用模型预测噪声，计算均方误差损失。
        """
        # 随机生成每个样本对应的时间步 t
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)

        # 生成与 x_0 形状相同的高斯噪声
        noise = torch.randn_like(x_0)
        # 根据时间步 t，给 x_0 添加噪声生成 x_t
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +  # 保留 x_0 的一部分
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)  # 添加噪声

        # 用模型预测 x_t 中的噪声，并与真实噪声进行对比，计算损失
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')

        return loss  # 返回损失


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model  # 训练好的扩散模型
        self.T = T  # 扩散过程的总步数

        # betas 是扩散过程中每个时间步的噪声强度，从 beta_1 线性插值到 beta_T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas  # alphas 是扩散过程中的保留因子，表示输入数据保留的比例
        alphas_bar = torch.cumprod(alphas, dim=0)  # alphas_bar 是 alphas 的累积乘积，用于表示从第 1 步到 t 步的总保留比例
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # 前一时间步的 alphas_bar，第一步 pad 为 1

        # coeff1 和 coeff2 是从噪声数据恢复原始数据时所需要的系数
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))  # 用于计算均值
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))  # 用于计算均值

        # posterior_var 是在采样过程中用于给模型预测添加噪声的方差
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
    
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        通过时间步 t 和模型预测的噪声 eps, 根据当前的加噪数据 x_t 预测前一个时间步的均值 xt_prev_mean
        """
        assert x_t.shape == eps.shape
        # 利用 coeff1 和 coeff2 来从 x_t 中去除预测的噪声 eps, 从而得到前一步的估计
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
    
    def p_mean_variance(self, x_t, t):
        """
        计算给定时间步 t 的 x_t, 它返回前一时间步的均值 xt_prev_mean 和方差 var
        """
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var
    
    def forward(self, x_T):
        """
        采样过程：从时间步 T 开始逐步逆向生成数据，最终得到无噪声的 x_0。
        """
        x_t = x_T  # 从最后时间步的 x_T（完全随机噪声）开始
        for time_step in reversed(range(self.T)):  # 从时间 T 到 0 逐步逆向处理
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step  # 当前时间步 t
            mean, var = self.p_mean_variance(x_t=x_t, t=t)  # 计算当前时间步的均值和方差

            # 当 time_step > 0 时，需要添加噪声；time_step == 0 时则不再加噪
            if time_step > 0:
                noise = torch.randn_like(x_t)  # 生成新的噪声
            else:
                noise = 0  # 最后一步不加噪声

            x_t = mean + torch.sqrt(var) * noise  # 根据均值和方差生成新的 x_t
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."  # 检查是否有 NaN 值
        x_0 = x_t  # 最终得到无噪声的 x_0
        return torch.clip(x_0, -1, 1)  # 将输出值限制在 [-1, 1] 范围内


