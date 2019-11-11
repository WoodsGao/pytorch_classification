import cv2
import random
import numpy as np


# bigger better
class PerspectiveProject:
    def __init__(self, rate=0.1, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        src = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], 0],
                          [img.shape[1], img.shape[0]]])
        dst = src + np.float32(
            (np.random.rand(4, 2) * 2 - 1) *
            np.float32([img.shape[0], img.shape[1]]) * self.rate)
        p_matrix = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, p_matrix, (img.shape[1], img.shape[0]))

        if det is not None:
            new_det = list()
            # detection n*(x1 y1 x2 y2 x3 y3 x4 y4)
            for i in range(4):
                point = np.concatenate(
                    [det[:, 2 * i:2 * i + 2],
                     np.ones([det.shape[0], 1])], 1)
                point = np.dot(p_matrix, point.transpose(1, 0)).transpose(1, 0)
                point[:, :2] /= point[:, 2:]
                new_det.append(point[:, :2])
            det = np.concatenate(new_det, 1)
        if seg is not None:
            seg = cv2.warpPerspective(seg, p_matrix,
                                      (seg.shape[1], seg.shape[0]))
        return img, det, seg


class HSV:
    def __init__(self, rate=[0.01, 0.7, 0.4]):
        self.rate = np.float32([[rate]])

    def __call__(self, img, det=None, seg=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = np.float32(img)
        img += img * self.rate
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img, det, seg


class Blur:
    def __init__(self, ksize=3):
        self.ksize = ksize

    def __call__(self, img, det=None, seg=None):
        img = cv2.blur(img, (self.ksize, self.ksize))
        return img, det, seg


class Pepper:
    def __init__(self, rate=0.05):
        self.rate = rate

    def __call__(self, img, det=None, seg=None):
        size = img.shape[0]
        amount = int(size * self.rate / 2 * random.random())

        x = np.random.randint(0, img.shape[0], amount)
        y = np.random.randint(0, img.shape[1], amount)
        img[x, y] = [0, 0, 0]

        x = np.random.randint(0, img.shape[0], amount)
        y = np.random.randint(0, img.shape[1], amount)
        img[x, y] = [255, 255, 255]
        return img, det, seg
