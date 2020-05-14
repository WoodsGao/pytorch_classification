import os
import os.path as osp
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from imgaug import augmenters as ia
from pytorch_modules.utils import IMG_EXT

TRAIN_AUGS = ia.SomeOf(
    [0, 3],
    [
        ia.WithColorspace(
            to_colorspace='HSV',
            from_colorspace='RGB',
            children=ia.Sequential([
                ia.WithChannels(
                    0,
                    ia.SomeOf([0, None],
                              [ia.Add((-10, 10)),
                               ia.Multiply((0.95, 1.05))],
                              random_state=True)),
                ia.WithChannels(
                    1,
                    ia.SomeOf([0, None],
                              [ia.Add((-50, 50)),
                               ia.Multiply((0.8, 1.2))],
                              random_state=True)),
                ia.WithChannels(
                    2,
                    ia.SomeOf([0, None],
                              [ia.Add((-50, 50)),
                               ia.Multiply((0.8, 1.2))],
                              random_state=True)),
            ])),
        ia.Dropout([0.015, 0.1]),  # drop 5% or 20% of all pixels
        ia.Sharpen((0.0, 1.0)),  # sharpen the image
        ia.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-0.1,
                   0.1)),  # rotate by -45 to 45 degrees (affects heatmaps)
        ia.ElasticTransformation(
            alpha=(0, 10),
            sigma=(0, 10)),  # apply water effect (affects heatmaps)
        ia.PiecewiseAffine(scale=(0, 0.03), nb_rows=(2, 6), nb_cols=(2, 6)),
        ia.GaussianBlur((0, 3)),
        ia.Fliplr(0.1),
        ia.Flipud(0.1),
        ia.LinearContrast((0.5, 1)),
        ia.AdditiveGaussianNoise(loc=(0, 10), scale=(0, 10))
    ],
    random_state=True)


class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_size=224, augments=TRAIN_AUGS, multi_scale=False, rect=False):
        super(ClsDataset, self).__init__()
        self.path = path
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert len(img_size) == 2
        self.img_size = img_size
        self.multi_scale = multi_scale
        self.rect = rect
        self.augments = augments
        self.data = []
        self.classes = []
        self.build_data()
        self.data.sort()

    def build_data(self):
        self.data = []
        data_dir = osp.dirname(self.path)
        class_names = [n for n in os.listdir(data_dir)]
        class_names = [
            cn for cn in class_names if osp.isdir(osp.join(data_dir, cn))
        ]
        class_names.sort()
        self.classes = class_names
        with open(self.path, 'r') as f:
            names = [n for n in f.read().split('\n') if n]
        self.data = [[
            osp.join(data_dir, name),
            self.classes.index(osp.basename(osp.dirname(name)))
        ] for name in names if osp.splitext(name)[1] in IMG_EXT]

    def get_item(self, idx):
        img = cv2.imread(self.data[idx][0])
        img = img[:, :, ::-1]
        h, w, c = img.shape

        if self.rect:
            scale = min(self.img_size[0] / w, self.img_size[1] / h)
            resize = ia.Sequential([
                ia.Resize({
                    'width': int(w * scale),
                    'height': int(h * scale)
                }),
                ia.PadToFixedSize(*self.img_size,
                                  pad_cval=[123.675, 116.28, 103.53],
                                  position='center')
            ])
        else:
            resize = ia.Resize({
                'width': self.img_size[0],
                'height': self.img_size[1]
            })
        img = resize.augment_image(img)
        # augment
        if self.augments is not None:
            augments = self.augments.to_deterministic()
            img = augments.augment_image(img)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.ByteTensor(img), self.data[idx][1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_item(idx)

    def post_fetch_fn(self, batch):
        imgs, labels = batch
        imgs = imgs.float()
        imgs -= torch.FloatTensor([123.675, 116.28,
                                   103.53]).reshape(1, 3, 1, 1).to(imgs.device)
        imgs /= torch.FloatTensor([58.395, 57.12,
                                   57.375]).reshape(1, 3, 1, 1).to(imgs.device)
        if self.multi_scale:
            h = imgs.size(2)
            w = imgs.size(3)
            scale = random.uniform(0.7, 1.5)
            h = int(h * scale / 32) * 32
            w = int(w * scale / 32) * 32
            imgs = F.interpolate(imgs, (h, w))
        return (imgs, labels.long())
