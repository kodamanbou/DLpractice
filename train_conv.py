import os
from scipy.misc import imresize
from optimizer import *
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from conv_net import DeepConvNet
from trainer import Trainer

save_file = '/dataset.pkl'


def scale_augmentation(image, scale_range=(256, 400), crop_size=224):
    scale_size = np.random.randint(*scale_range)
    image = imresize(image, (scale_size, scale_size))
    image = random_crop(image, (crop_size, crop_size))

    return image.transpose(2, 0, 1)  # 軸の入れ替え.


def random_crop(image, crop_size=(224, 224)):
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    return image


def load_images(dir):
    filenames = [os.getcwd() + dir + '/' + filename
                 for filename in os.listdir(os.getcwd() + dir)
                 if not filename.startswith('.')]  # そういうとこやぞ！Mac！！(.DB_Store対策)
    images = []

    for filename in filenames:
        try:
            print(filename)
            image = np.array(Image.open(filename), dtype=np.float32)
            image = scale_augmentation(image)
            images.append(image)
        except:
            print('LoadError')

    return images


def _change_one_hot_label(X):
    T = np.zeros((X.size, 3))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def init_dataset():
    dataset = {}

    dog_list = np.concatenate(load_images('/dog'), axis=0)
    cat_list = np.concatenate(load_images('/cat'), axis=0)
    person_list = np.concatenate(load_images('/person'), axis=0)

    X = np.concatenate([dog_list, cat_list, person_list], axis=0)
    X = X.reshape(-1, 3, 224, 224)

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

    # (訓練画像, 訓練ラベル), (テスト画像, テストラベル)に分ける.
    return train_test_split(dataset['img'], dataset['label'], test_size=0.3)


# データセット読み込み.
X_train, X_test, t_train, t_test = load_dataset()

network = DeepConvNet()
trainer = Trainer(network, X_train, t_train, X_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

network.save_params('deep_convnet_params.pkl')
print('All done !!')
