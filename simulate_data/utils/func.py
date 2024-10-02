"""
These func were derived by @author: Xianming Wu
"""
import  numpy as np
from  math import pi
from numpy import fft
import cv2
import matplotlib.pyplot as plt

def overlap_rate(r,delta):
    overlap=(2*r**2*np.arctan(np.sqrt(r*r-delta**2/4)/(0.5*delta))\
              - delta*np.sqrt(r**2-0.25*delta**2)) / (3.14*r**2)
    return overlap

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
def SNR(image, noise_image):
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

def show_object_and_probe(probe, obj_am, obj_ph, output_path='object_and_probe.png'):
    # show object and probe
    fig, ax = plt.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0.5})
    plt.suptitle('object and probe')

    ax = ax.flatten()
    ax0 = ax[0].imshow(obj_am, cmap='gray')
    ax[0].set_title('object amplitude')
    ax1 = ax[1].imshow(obj_ph, cmap='gray')
    ax[1].set_title('object phase')
    ax[2].imshow(np.abs(probe) / np.max(np.abs(probe)), cmap='gray')
    ax[2].set_title('probe amplitude')
    ax[3].imshow(np.angle(probe), cmap='gray')
    ax[3].set_title('probe phase')

    fig.colorbar(ax0, ax=ax[0])
    fig.colorbar(ax1, ax=ax[1])

    # Save the figure instead of showing it
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存

def imcrop(image,pixsum=100):

    [row, col] = np.shape(image)
    x_center = col // 2
    y_center = row // 2
    abstract = image[(y_center - pixsum // 2) : (y_center + pixsum // 2),
                (x_center - pixsum // 2): (x_center + pixsum // 2)]

    return abstract

def lowPassFiltering(img,size):#传递参数为傅里叶变换后的频谱图和滤波尺寸
    h, w = img.shape[0:2]#获取图像属性
    h1,w1 = int(h/2), int(w/2)#找到傅里叶频谱图的中心点
    img2 = np.zeros((h, w), dtype=float)#定义空白黑色图像，和傅里叶变换传递的图尺寸一致
    img2[h1-int(size/2):h1 +int(size/2), w1-int(size/2):w1+int(size/2)] = 1#中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为1，保留低频部分
    img3=img2*img #将定义的低通滤波与传入的傅里叶频谱图一一对应相乘，得到低通滤波
    return img3