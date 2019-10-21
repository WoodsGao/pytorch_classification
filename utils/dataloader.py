from .cv_utils import dataloader
import os
import numpy as np
import cv2


class Dataloader(dataloader.Dataloader):
    def build_data_list(self):
        self.classes = os.listdir(self.path)
        self.classes.sort()
        for ci, c in enumerate(self.classes):
            names = os.listdir(os.path.join(self.path, c))
            for name in names:
                target = np.zeros(len(self.classes))
                target[ci] = 1
                self.data_list.append(
                    [os.path.join(self.path, c, name), target])

    def worker(self, message):
        img = cv2.imread(message[0])
        img = cv2.resize(img, (self.img_size, self.img_size))
        for aug in self.augments:
            img, _, __ = aug(img)
        return img, message[1]