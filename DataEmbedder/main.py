from imageio import imread
import numpy as np
import random
import matplotlib.pyplot as plt
import cmath
import os

random.seed(3)


def get_img(path):
    return imread(path)


def byte2bit(matrix):
    matrix = np.array(matrix)
    m, n = matrix.shape
    new_matrix = np.zeros((8, m, n), dtype=np.int)
    for k in range(8):
        new_matrix[k] = (matrix / pow(2, k)) % 2
    return new_matrix


def bit2byte(matrix):
    matrix = np.array(matrix)
    bit_num, m, n = matrix.shape
    new_matrix = np.zeros((m, n), dtype=np.int)
    for k in range(8):
        new_matrix += matrix[k] * pow(2, k)
    return new_matrix


def get_encryption_key(k, stream_len):
    # 采用RC4算法生成流密码
    s = [x for x in range(256)]
    t = [k[x % len(k)] for x in range(256)]
    j = 0
    for i in range(256):
        j = (j+s[i]+t[i]) % 256
        temp = s[i]
        s[i] = s[j]
        s[j] = temp
    key_stream = []
    m = n = 0
    for i in range(stream_len):
        m = (m + 1) % 256
        n = (n + s[n]) % 256
        temp = s[m]
        s[m] = s[n]
        s[n] = temp
        idx = (s[m] + s[n]) % 256
        key_stream.append(s[idx])
    return key_stream


def image_encryption(image, short_key):
    image_size = len(image)
    # 将图像处理成为bit表示
    image = byte2bit(image)
    # 通过RC4获得encryption key
    key = get_encryption_key(short_key, image_size*image_size)
    key = np.array(key).reshape(image_size, image_size)
    key = byte2bit(key)
    # 明文和随机字节做异或运算
    encrypted_image = image ^ key
    # 将图像转为byte表示
    encrypted_image = bit2byte(encrypted_image)
    return encrypted_image


def get_data_hidding_key(key_num, key_range, key_len):
    # 获得data hiding key
    key = [None] * key_num
    for i in range(key_num):
        key[i] = random.sample(range(key_range), key_len)
    return key


def data_embedding(image, data_hidding_key, datas=None):
    # 将数据嵌入到每一块当中
    image = byte2bit(image)
    for idx, data in enumerate(datas):
        m = int(idx / block_num)
        n = int(idx % block_num)
        id = 0
        for i in range(m*block_size, (m+1)*block_size):
            for j in range(n*block_size, (n+1)*block_size):
                if (data == 0 and id not in data_hidding_key[idx]) or (data == 1 and id in data_hidding_key[idx]):
                    for k in range(3):
                        image[k][i][j] = (image[k][i][j] + 1) % 2
                id += 1
    image = bit2byte(image)
    return image


def data_extraction(image, data_hidding_key):
    image_len = len(image)
    old_image = image.copy()
    image = byte2bit(image.copy())
    # 翻转S0/1构造两个新的图像
    flip_zero_image = image.copy()
    flip_one_image = image.copy()
    for i in range(image_len):
        for j in range(image_len):
            block_i = int(i / block_size)
            block_j = int(j / block_size)
            block_id = int(block_i * block_num + block_j)
            pixel_id = int((i - block_i*block_size) * block_size + j - block_j*block_size)
            if pixel_id in data_hidding_key[block_id]:
                for k in range(3):
                    flip_one_image[k][i][j] = (flip_one_image[k][i][j] + 1) % 2
            else:
                for k in range(3):
                    flip_zero_image[k][i][j] = (flip_zero_image[k][i][j] + 1) % 2
    # 根据像素与周围的联系程度，判断哪一个翻转之后的图像是正确的
    flip_zero_image = bit2byte(flip_zero_image)
    flip_one_image = bit2byte(flip_one_image)
    f1_map = np.empty(flip_one_image.shape)
    f0_map = np.empty(flip_zero_image.shape)
    for i in range(1, image_len-1):
        for j in range(1, image_len-1):
            f1_map[i][j] = abs(flip_one_image[i][j] - (flip_one_image[i-1][j] + flip_one_image[i][j-1] +
                                                       flip_one_image[i+1][j] + flip_one_image[i][j+1]) / 4)
            f0_map[i][j] = abs(flip_zero_image[i][j] - (flip_zero_image[i-1][j] + flip_zero_image[i][j-1] +
                                                       flip_zero_image[i+1][j] + flip_zero_image[i][j+1]) / 4)
    # 更新旧图像，提取数据
    data = []
    for m in range(block_num):
        for n in range(block_num):
            f1 = f0 = 0
            for u in range(1, block_size-1):
                for v in range(1, block_size-1):
                    pixel_i = m*block_size + u
                    pixel_j = n*block_size + v
                    f1 += f1_map[pixel_i][pixel_j]
                    f0 += f0_map[pixel_i][pixel_j]
            x1 = m * block_size
            x2 = (m + 1) * block_size
            y1 = n * block_size
            y2 = (n + 1) * block_size
            if f1 > f0:
                data.append(0)
                old_image[x1:x2][y1:y2] = flip_zero_image[x1:x2][y1:y2]
            else:
                data.append(1)
                old_image[x1:x2][y1:y2] = flip_one_image[x1:x2][y1:y2]
    return old_image, data


def show_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def get_error_rate(true_lst, lst):
    true_lst = np.array(true_lst)
    lst = np.array(lst)
    equal = (true_lst == lst)
    return float(1 - equal.sum()/equal.shape)


def get_psnr(true_img, image):
    true_img = np.array(true_img)
    image = np.array(image)
    m, n = true_img.shape
    diff = (true_img - image) * (true_img - image)
    mse = diff.sum() / (m * n)
    return 10 * cmath.log(255*255 / mse, 10)


def get_ec(img_len, block_size):
    additional_bit = pow(int(img_len / block_size), 2)
    return additional_bit / (img_len*img_len)


def get_paths(root):
    g = os.walk(root)
    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            files.append(os.path.join(path, file_name))
    return files


if __name__ == '__main__':
    # 程序参数
    root = './data/'
    short_key_len = 16
    block_sizes = [8, 16, 32, 64]
    # 绘图参数
    markers = ['o', 'd', 'x', 's', '.', '*', '+', 'H', 'p']
    # 预处理
    paths = get_paths(root)
    image_names = [path[len(root):-4] for path in paths]

    extracted_bit_error_rates = []
    ecs = []
    psnrs = []
    for path in paths:
        print(path)
        image_error_rate = []
        image_ecs = []
        image_psnrs = []
        for block_size in block_sizes:
            img = get_img(path)  # 图像读取
            block_num = int(len(img) / block_size)
            encryption_key = [random.randrange(256) for x in range(short_key_len)]  # 生成encryption key
            data_hidding_key = get_data_hidding_key(int(block_num*block_num),
                                                    int(block_size*block_size), int(block_size*block_size / 2))
            data = [random.randrange(0, 2, 1) for x in range(int(block_num * block_num))]  # 生成需要隐藏的数据

            # show_img(img)
            # 图像加密
            encrypted_img = image_encryption(img, encryption_key)
            # show_img(encrypted_img)
            # 图像嵌入数据
            embedded_img = data_embedding(encrypted_img, data_hidding_key, data)
            # show_img(embedded_img)
            # 图像解密
            decrypted_img = image_encryption(embedded_img, encryption_key)
            # show_img(decrypted_img)
            # 数据提取，原始图像恢复
            origial_img, extracted_data = data_extraction(decrypted_img, data_hidding_key)
            # show_img(origial_img)
            image_error_rate.append(get_error_rate(data, extracted_data))
            image_psnrs.append(get_psnr(img, origial_img))
            image_ecs.append(get_ec(len(img), block_size))
        extracted_bit_error_rates.append(image_error_rate)
        psnrs.append(image_psnrs)
        ecs.append(image_ecs)
    # 绘制block_size - extracted bits error rate曲线
    for idx, image_er in enumerate(extracted_bit_error_rates):
        plt.plot(block_sizes, image_er, marker=markers[idx], label=image_names[idx])
    plt.xlabel('Side length of each block s')
    plt.ylabel('Extracted-bit error rate (%)')
    plt.title('Extracted-bit error rate with respect to block sizes.')
    plt.legend()
    plt.show()
    # 绘制Embedding capacity - psnr曲线
    for idx, psnr in enumerate(psnrs):
        ec = ecs[idx]
        plt.plot(ec, psnr)
        plt.xlabel('Embedding Capacity (bpp)')
        plt.ylabel('PSNR (dB)')
        plt.title(image_names[idx])
        plt.show()
    # 输出结果以保存
    with open('result.txt', 'w') as f:
        f.write('block sizes: ')
        f.write(str(block_sizes))
        f.write('\nExtracted-bit error rate: ')
        for idx, er in enumerate(extracted_bit_error_rates):
            f.write('\n')
            f.write(str(image_names[idx]))
            f.write(': ')
            f.write(str(er))
        f.write('\nEmbedding Capacity (bpp): ')
        for idx, ec in enumerate(ecs):
            f.write('\n')
            f.write(str(image_names[idx]))
            f.write(': ')
            f.write(str(ec))
        f.write('\nPSNR (dB)')
        for idx, psnr in enumerate(psnrs):
            f.write('\n')
            f.write(str(image_names[idx]))
            f.write(': ')
            f.write(str(psnr))