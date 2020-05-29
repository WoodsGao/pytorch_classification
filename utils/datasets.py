import os
import os.path as osp
import random

import cv2
import imgaug as ia
import numpy as np
import torch
import torch.nn.functional as F
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from pytorch_modules.utils import IMG_EXT

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

TRAIN_AUGS = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(
            iaa.CropAndPad(
                percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))),
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.8, 1.2),
                    "y": (0.8, 1.2)
                },  # scale images to 80-120% of their size, individually per axis
                translate_percent={
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2)
                },  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[
                    0,
                    1
                ],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(
                    0,
                    255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.
                ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf(
            (0, 5),
            [
                sometimes(
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
                ),  # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur(
                        (0,
                         3.0)),  # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(
                        k=(2, 7)
                    ),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(
                        k=(3, 11)
                    ),  # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0),
                            lightness=(0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.BlendAlphaSimplexNoise(
                    iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                               direction=(0.0, 1.0)),
                    ])),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255),
                    per_channel=0.5),  # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5
                                ),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15),
                                      size_percent=(0.02, 0.05),
                                      per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                iaa.Add(
                    (-10, 10), per_channel=0.5
                ),  # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation(
                    (-20, 20)),  # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.BlendAlphaFrequencyNoise(
                        exponent=(-4, 0),
                        foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                        background=iaa.LinearContrast((0.5, 2.0)))
                ]),
                iaa.LinearContrast(
                    (0.5, 2.0),
                    per_channel=0.5),  # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),  # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(
                    0.01, 0.05))),  # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True)
    ],
    random_order=True)


class ClsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 img_size=224,
                 augments=TRAIN_AUGS,
                 multi_scale=False,
                 rect=False):
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
            resize = iaa.Sequential([
                iaa.Resize({
                    'width': int(w * scale),
                    'height': int(h * scale)
                }),
                iaa.PadToFixedSize(*self.img_size,
                                   pad_cval=[123.675, 116.28, 103.53],
                                   position='center')
            ])
        else:
            resize = iaa.Resize({
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
