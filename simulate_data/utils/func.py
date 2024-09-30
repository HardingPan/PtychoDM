import  numpy as np
from  math import pi
from numpy import fft
import cv2
import matplotlib.pyplot as plt

def overlap_rate(r,delta):
    # This formula is derived by @author: Xianming Wu
    overlap=(2*r**2*np.arctan(np.sqrt(r*r-delta**2/4)/(0.5*delta))\
              - delta*np.sqrt(r**2-0.25*delta**2)) / (3.14*r**2)
    return overlap


def set_pinhole(probe_config):
   """创建一个二维探针孔径, 用于模拟一个具有给定半径的圆形孔"""
   m, n, r = probe_config['m'], probe_config['n'], probe_config['r']

   pinhole = np.zeros((m, n))
   x = np.array(np.arange(0, m, 1), np.int32)
   y = np.array(np.arange(0, n, 1), np.int32)
   j, i =  np.meshgrid(x, y)
   i = np.reshape(i, -1)
   j = np.reshape(j, -1)
   for p in zip(i,j):
     if (p[0]-m/2)**2 + (p[1]-n/2)**2 <= r**2:
         pinhole[p[0],p[1]]=1

   return pinhole

def set_probe_and_obj(probe_am, obj_am, obj_ph):
    #judge image chanels,if it is not single chanels ,translating it
    probe_am = convert_to_grayscale(probe_am)
    obj_am = convert_to_grayscale(obj_am)
    obj_ph = convert_to_grayscale(obj_ph)
    
    #pad image ,adjust object's size to target value
    target_obj = 400
    target_probe = 64
    (n1, n2) = probe_am.shape
    (m1, m2) = obj_am.shape
    (q1, q2) = obj_ph.shape
    probe_am = np.pad(probe_am, (((target_probe-n1)//2,(target_probe-n1)//2),\
                                 ((target_probe-n2)//2,(target_probe-n2)//2)),\
                                  'constant', constant_values=(0,0))
    obj_am = np.pad(obj_am, (((target_obj-m1)//2,(target_obj-m1)//2),\
                             ((target_obj-m2)//2,(target_obj-m2)//2)),\
                               'constant', constant_values=(0,0))
    obj_ph = np.pad(obj_ph, (((target_obj-q1)//2,(target_obj-q1)//2),\
                             ((target_obj-q2)//2,(target_obj-q2)//2)),\
                              'constant', constant_values=(0,0))
    #normlize amplitude to 1
    obj_am = Normlize(obj_am,0,1)
    obj_ph = Normlize(obj_ph,-pi, pi)
    probe_am = Normlize(probe_am,0, 1)

    print('probe size', probe_am.shape,'\nobject size', obj_am.shape)

    return probe_am, obj_am, obj_ph

def propagate(input, propagator, ptycho_cfg):
    """
    模拟波前的传播
    input: 输入的波前（通常是复数形式的振幅和相位信息）
    propagator: 传播方法，支持 'Fourier' 和 'Fresnel' 两种方式
    dx: 输入波前的像素间距，即像素的物理尺寸
    wavelength: 光波的波长，默认为 632.8nm
    z: 要传播的距离，即波前传播的物理距离
    """
    dx = ptycho_cfg['pix']
    wavelength = ptycho_cfg['lambda']
    z = ptycho_cfg['z1']
    ysize, xsize = np.shape(input)
    x = np.array(np.arange(-xsize/2, xsize/2, 1))
    y = np.array(np.arange(-ysize/2, ysize/2, 1))
    fx= x/(dx*xsize)
    fy= y/(dx*ysize)
    fx,fy = np.meshgrid(fx, fy)

    if propagator=='Fourier':
        if  z > 0:
            output=fft.fftshift(fft.fft2(input))
        elif z==0:
            output=input
        else:
            output=fft.ifft2(fft.ifftshift(input))

    # Calculate approx phase distribution for each plane wave component
    elif propagator=='Fresnel':
        w=fx**2+fy**2
        F=fft.fftshift(fft.fft2(input))
        output=fft.ifft2(fft.ifftshift(F*np.exp(-1j*pi*z*wavelength*w)))
    else:
         output=input
         print('invalid propagate way,please change')

    return output

def ccd_intensities(object, probe, ptycho_cfg, propagator):
    """
    计算在给定探针和物体的情况下, 经过一定传播距离后, 在 CCD 上获得的衍射图像强度
    # object -- 物体的全貌
    # probe -- 探针的大小（其大小与衍射图像相同）
    # lamda -- 波长
    # origin -- 采样起始点
    # Nx -- x方向上的采样点数量
    # Ny -- y方向上的采样点数量
    # delta -- 采样间隔
    # pix -- CCD 的像素大小
    # z -- 传播距离
    # propagator -- 传播方式（如傅里叶或菲涅尔）
    """
    lamda = ptycho_cfg['lambda']
    origin = ptycho_cfg['origin']
    Nx, Ny = ptycho_cfg['Nx'], ptycho_cfg['Ny']
    delta, pix, z = ptycho_cfg['delta'], ptycho_cfg['pix'], ptycho_cfg['z2']

    positions = np.zeros((Nx * Ny, 2), dtype=np.int32)  # 初始化位置数组
    dy = delta  # y方向的间隔
    dx = delta  # x方向的间隔

    # 计算采样位置
    positions[:, 1] = np.tile(np.arange(Nx) * dx, Ny)  # x方向的位置
    positions[:, 0] = np.repeat(np.arange(Ny) * dy, Nx)  # y方向的位置
    positions += origin  # 加上起始偏移

    positions2 = positions.copy()  # 复制位置数组
    random_offet = np.random.randint(-7, 7, np.shape(positions))  # 添加随机偏移
    positions += random_offet  # 更新位置

    diffset = []  # 存储 CCD 强度的列表
    sum = 0  # 初始化总和

    # 照射区域的索引
    illuindy, illuindx = np.indices((probe.shape))  # 创建采样范围的索引

    for pos in positions:
        # 传播到远场并计算绝对值平方作为衍射强度
        img, sum = poisson_noise(
            abs(
                propagate(object[pos[0] + illuindy, pos[1] + illuindx] * probe, propagator, ptycho_cfg)
            ) ** 2,
            alpha=1, sum=sum
        )
        diffset.append(abs(img))  # 存储强度图像
    # print('Diffraction pattern created')  # 打印信息表示已创建衍射图样
    # cv2.imwrite("output.png", diffset[0])

    return diffset, positions, illuindx, illuindy  # 返回强度图像、位置及照射索引

def poisson_noise(image, alpha, sum):
   [rows, cols] = image.shape
   image_noise = image + np.random.normal(loc=0, scale = alpha*np.sqrt(image), size=(rows, cols))
#    print(SNR(image,image_noise))
   sum = sum + SNR(image,image_noise)
#    print(sum)

   return image_noise,sum

def convert_to_grayscale(image):
    if image.ndim == 3 and image.shape[2] == 3:  # 如果是三通道图像
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    return image  # 如果已经是灰度图像，直接返回

def Normlize(img, a, b):

    min = np.min(img)
    max = np.max(img)
    Norm_img = (img-min) / (max - min ) * (b -a)

    return Norm_img

# SNR caculate
def SNR(image ,noise_image ):
     image = image / np.max(image)
     noise_image = noise_image / np.max(noise_image)
     image2 = image - noise_image
     image1 = image
     image11 = image1 ** 2
     image22 = image2 ** 2
     p = np.sum(image11 )
     d = np.sum(image22)
     snr = -10 * np.log10(d / p)
     return snr