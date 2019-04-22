from pytube import YouTube
import os
from tqdm import tqdm

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'urls.txt')
    with open(path) as f:
        for url in tqdm(f):
            try:
                yt = YouTube(url)
                yt.streams.filter(subtype='mp4').first().download('./videos')
                print('Downloaded: {}'.format(url))
            except:
                print('Download Error !')
