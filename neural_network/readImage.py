import numpy as np

imagefilepath = './dataset/train-images-idx3-ubyte'
labelfilepath = './dataset/train-labels-idx1-ubyte'


def read_image(filepath, num):
    with open(filepath, 'rb') as f:
        f.seek(16)
        data = f.read(num*28*28)
        images = np.empty(num*28*28, dtype=np.float)
        for i in range(num*28*28):
            images[i] = data[i] / 256
    return images.reshape(-1, 28*28)


def read_label(filepath, num):
    with open(filepath, 'rb') as f:
        f.seek(8)
        data = f.read(num)
        label = np.empty(num, dtype=np.int64)
        for i in range(num):
            label[i] = data[i]
    return label
