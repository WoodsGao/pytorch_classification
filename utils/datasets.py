import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
from random import randint
from threading import Thread
from . import config


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 cache_dir=None,
                 cache_len=3000,
                 img_size=224,
                 augments=[]):
        self.path = path
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = True
        else:
            self.cache = False
        self.img_size = img_size
        self.augments = augments
        self.data = []
        data_dir = os.path.dirname(self.path)
        class_names = [n for n in os.listdir(data_dir)]
        class_names = [cn for cn in class_names if os.path.isdir(os.path.join(data_dir, cn))]
        self.classes = class_names
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        self.data = [[
            name,
            self.classes.index(os.path.basename(os.path.dirname(name)))
        ] for name in names if os.path.splitext(name)[1] in config.IMG_EXT]
        if self.cache:
            self.counts = [-1 for i in self.data]
            self.cache_path = [
                os.path.join(cache_dir, str(i)) for i in range(len(self.data))
            ]
            self.cache_len = cache_len
            self.cache_memory = [None for i in range(cache_len)]

    def refresh_cache(self, idx):
        item = self.get_item(idx)
        if idx < self.cache_len:
            self.cache_memory[idx] = item
        else:
            torch.save(item, self.cache_path[idx])
        self.counts[idx] = 0
        return item

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        img = cv2.resize(img, (self.img_size, self.img_size))
        for aug in self.augments:
            img, _, __ = aug(img)
        return torch.FloatTensor(img), self.data[idx][1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.cache:
            return self.get_item(idx)
        if self.counts[idx] < 0:
            item = self.refresh_cache(idx)
            return item
        self.counts[idx] += 1
        if idx < self.cache_len:
            item = self.cache_memory[idx]
        else:
            item = torch.load(self.cache_path[idx])
        if self.counts[idx] > randint(3, 15):
            t = Thread(target=self.refresh_cache, args=(idx, ))
            t.setDaemon(True)
            t.start()
        return item


def show_batch(save_path, inputs, targets, classes):
    imgs = []
    for bi, (img, c) in enumerate(zip(inputs, targets)):
        img *= 255.
        img.clamp(0, 255)
        img = img.long().numpy().transpose(1, 2, 0)
        img = img[:, :, ::-1]
        imgs.append(img)

    imgs = np.concatenate(imgs, 1)
    save_img = imgs
    cv2.imwrite(save_path, save_img)
