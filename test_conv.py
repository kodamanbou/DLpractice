import numpy as np
import pickle
from PIL import Image
from conv_net import DeepConvNet
import os
from scipy.misc import imresize


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


path = input('Image Directory ->')
image = np.array(Image.open(path), dtype=np.float32)
image = imresize(image, (227, 227))
X = image.transpose(2, 0, 1).reshape(1, 3, 227, 227).astype(np.float32)
X /= 255.0
t = ['dog', 'cat', 'person']

network = DeepConvNet()
network.load_params(file_name='deep_convnet_params.pkl')
output = softmax(network.predict(X))
print(output)
label = t[np.argmax(output)]
print(label)
