import os
from scipy.misc import imresize
from optimizer import *
from PIL import Image
import pickle
import random
from sklearn.model_selection import train_test_split
from conv_net import DeepConvNet
from trainer import Trainer

save_file = '/dataset.pkl'


def cutout(image_origin, mask_size):
    image = np.copy(image_origin)
    image = image.transpose(1, 2, 0)
    mask_value = image.mean()

    h, w, _ = image.shape

    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_value)
    return image.transpose(2, 0, 1)


def scale_augmentation(image, scale_range=(256, 400), crop_size=227):
    scale_size = np.random.randint(*scale_range)
    image = imresize(image, (scale_size, scale_size))
    image = random_crop(image, (crop_size, crop_size))

    return image.transpose(2, 0, 1)  # 軸の入れ替え.


def random_crop(image, crop_size=(227, 227)):
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    return image


def load_images(dir, max):
    filenames = [os.getcwd() + dir + '/' + filename
                 for filename in os.listdir(os.getcwd() + dir)
                 if not filename.startswith('.')]  # そういうとこやぞ！Mac！！(.DB_Store対策)
    images = []

    for filename in filenames:
        try:
            print(filename)
            image = np.array(Image.open(filename), dtype=np.float32)

            if len(image.shape) < 3:
                continue
            if image.shape[2] != 3:
                continue

            image = scale_augmentation(image)
            images.append(image)
        except:
            print('LoadError')

    # Data augmentation.
    if len(images) < max:
        sample = random.choices(images, k=(max - len(images)))
        for img in sample:
            img = cutout(img, 110)
            images.append(img)

    # debug.
    print(len(images))

    return images


def _change_one_hot_label(X):
    T = np.zeros((X.size, 3))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def init_dataset():
    dataset = {}

    dog_list = np.concatenate(load_images('/dog', 3000), axis=0)
    cat_list = np.concatenate(load_images('/cat', 3000), axis=0)
    person_list = np.concatenate(load_images('/person', 3000), axis=0)

    # debug.
    print(dog_list.shape)
    print(cat_list.shape)
    print(person_list.shape)

    X = np.concatenate([dog_list, cat_list, person_list], axis=0)
    X = X.reshape(-1, 3, 227, 227)

    t = np.concatenate([np.zeros(int(dog_list.shape[0] / 3)),
                        np.ones(int(cat_list.shape[0] / 3)),
                        np.full(int(person_list.shape[0] / 3), 2)],
                       axis=0)

    # データセットをシャッフル.
    for l in [X, t]:
        np.random.seed(1)
        np.random.shuffle(l)

    dataset['img'] = X
    dataset['label'] = t

    with open(os.getcwd() + save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print('Done')


def load_dataset(normalize=True, one_hot_label=False):
    if not os.path.exists(os.getcwd() + save_file):
        print('Initializing')
        init_dataset()

    with open(os.getcwd() + save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 正規化.
    if normalize:
        dataset['img'] = dataset['img'].astype(np.float32)
        dataset['img'] /= 255.0

    # 必要があればワンホットベクトルに変換.
    if one_hot_label:
        dataset['label'] = _change_one_hot_label(dataset['label'])

    dataset['label'] = dataset['label'].astype(np.int64)
    print(len(dataset['label']))

    # (訓練画像, 訓練ラベル), (テスト画像, テストラベル)に分ける.
    return train_test_split(dataset['img'], dataset['label'], test_size=0.3)


# データセット読み込み.
X_train, X_test, t_train, t_test = load_dataset()

network = DeepConvNet()
trainer = Trainer(network, X_train, t_train, X_test, t_test,
                  epochs=30, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
print('training start.')
trainer.train()

network.save_params('deep_convnet_params.pkl')
print('All done !!')
