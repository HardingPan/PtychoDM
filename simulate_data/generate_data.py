"""
生成含有振幅和相位的仿真数据
"""
import matplotlib.pyplot as plt
import numpy as  np
import time
from numpy import  fft
from  math import pi
import cv2
import random
from tqdm import tqdm
import os

# 仿真图像和探针的配置
from config import *
from utils.func import *
from utils.sim_probe import *

image_size = 256
cropsize = 64
overlap = overlap_rate(probe_config['r'], ptycho_config['delta'])
print(f"config of probe: {probe_config}")
print("overlap_rate is" , overlap)

class SimGenerator():
    def __init__(self, probe_cfg=probe_config, ptycho_cgf=ptycho_config) -> None:
        self.ptycho_cfg = ptycho_cgf
        # create pinhole
        self.pinhole = set_pinhole(probe_cfg)

    def set_pinhole(self, pinhole):
        # 将探针设置为自定义二维数组
        self.pinhole = pinhole

    def generate_single_obj(self, obj_am_path, obj_ph_path):
        obj_am = cv2.imread(obj_am_path, cv2.IMREAD_GRAYSCALE)
        obj_ph = cv2.imread(obj_ph_path, cv2.IMREAD_GRAYSCALE)
        probe_am = self.pinhole
        #judge image chanels, pad image and normlize amplitude to 1
        probe_am, obj_am, obj_ph = set_probe_and_obj(probe_am, obj_am, obj_ph)
        # 将幅度和相位结合起来, 生成一个复数形式的探针信号
        probe = probe_am * np.exp(1*1j)
        obj = obj_am * np.exp(obj_ph*1j)
        # 对probe进行前向传播
        probe = propagate(probe, 'Fresnel', self.ptycho_cfg)
        # show_object_and_probe(probe, obj_am, obj_ph)

        # get diffractions image, positions, illuindx, illuindy
        diffset, positions, illuindx, illuindy = ccd_intensities(obj, probe, self.ptycho_cfg, 'Fourier')
        # 形成初步的探针重构图像
        probe_r = np.zeros(diffset[2].shape)
        # 所有衍射图像的平方根的总和
        for i in range(len(diffset)):
            diffset[i] = imcrop(diffset[i], pixsum=cropsize)
            probe_r += np.sqrt(diffset[i])

        # 去掉 obj_am 和 obj_ph 文件名中的扩展名
        obj_am_name = os.path.splitext(os.path.basename(obj_am_path))[0]
        obj_ph_name = os.path.splitext(os.path.basename(obj_ph_path))[0]

        # 格式化保存文件名
        save_filename = f"AM_{obj_am_name}_PH_{obj_ph_name}.npy"

        # 保存 diffset 为 .npy 文件
        np.save(save_filename, diffset)

        # difine object shape(needed for reconstruction)
        probe_r = lowPassFiltering(probe_r, 17)
        probe_r = fft.ifft2(probe_r) / len(diffset)
        probe_r = abs(fft.ifftshift(probe_r))
        probe_r = probe_r / np.max(probe_r)
        



if __name__ == '__main__':
    generator = SimGenerator()
    obj_am = 'simulate_data/image_data/test_data/airplane256.png'
    obj_ph = 'simulate_data/image_data/test_data/boat256.png'
    generator.generate_single_obj(obj_am, obj_ph)



        
        

    

