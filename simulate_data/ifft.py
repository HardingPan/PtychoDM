import numpy as np
import cv2

def inverse_fourier_transform(image):
    """
    对输入图像进行傅立叶逆变换。

    参数:
    image -- 输入的复数频域图像

    返回:
    output -- 逆变换后的图像
    """
    # 进行傅立叶逆变换
    output = np.fft.ifft2(np.fft.ifftshift(image))
    # 获取实部并返回
    return np.abs(output)

if __name__ == '__main__':
    fft_img = cv2.imread('/Users/panding/workspace/PtychoDM/diffset.png', cv2.IMREAD_GRAYSCALE)
    output = inverse_fourier_transform(fft_img)
    print(output)
    cv2.imwrite('ifft.png', output)