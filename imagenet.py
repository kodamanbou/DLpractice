import sys
import os
from PIL import Image
from urllib import request


def download(url, decode=False):
    response = request.urlopen(url)
    if response.geturl() == 'https://s.yimg.com/pw/images/en-us/' \
                            'photo_unavailable.png':
        raise Exception("This photo is unavailable !!")

    body = response.read()
    if decode:
        body = body.decode()

    return body


def write(path, img):
    with open(path, 'wb') as f:
        f.write(img)


classes = {'cat': 'n02120997', 'dog': 'n02083346', 'person': 'n00007846'}
offset = 0
max = 2000

for dir, id in classes.items():
    print(dir)
    os.makedirs(dir, exist_ok=True)
    urls = download("http://www.image-net.org/api/text/imagenet.synset.geturls?"
                    "wnid=" + id, decode=True).split()
    print(str(len(urls)) + "枚ダウンロードするよ.")

    for i, url in enumerate(urls):
        if i < offset:
            continue
        if i > max:
            break

        try:
            file = os.path.split(url)[1]
            path = dir + '/' + file
            write(path, download(url))
            print('done: ' + file)
        except:
            print("Error has occured")
