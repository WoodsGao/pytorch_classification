import argparse
import os
import os.path as osp

import torch

from pytorch2caffe import pytorch2caffe
from models import MobileNetV2
from pytorch_modules.utils import fuse


def export2caffe(weights, num_classes, img_size):
    model = MobileNetV2(num_classes)
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights['model'])
    model.eval()
    fuse(model)
    name = 'MobileNetV2'
    dummy_input = torch.ones([1, 3, img_size[1], img_size[0]])
    pytorch2caffe.trans_net(model, dummy_input, name)
    pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str)
    parser.add_argument('-nc', '--num-classes', type=int, default=21)
    parser.add_argument('-s',
                        '--img_size',
                        type=int,
                        nargs=2,
                        default=[224, 224])
    opt = parser.parse_args()

    export2caffe(opt.weights, opt.num_classes, opt.img_size)
