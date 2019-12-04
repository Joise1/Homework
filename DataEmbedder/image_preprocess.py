import numpy as np
from imageio import imread
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(rgb, img_gray):
    # 原图 R G 通道不变，B 转换回彩图格式
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = (img_gray - 0.299 * R - 0.587 * G) / 0.114

    gray_rgb = np.zeros(rgb.shape)
    gray_rgb[:, :, 2] = B
    gray_rgb[:, :, 0] = R
    gray_rgb[:, :, 1] = G

    return gray_rgb


if __name__ == '__main__':
    rgb_path = "./data/Tank.tiff"
    rgb = imread(rgb_path)
    img = rgb2gray(rgb)
    img = np.array(img, dtype='uint8')
    img = Image.fromarray(img)
    img.save('./data/Tank.bmp', 'bmp')
