import argparse
import os
import os.path as osp

import torch

from models import MobileNetV2
from pytorch_modules.utils import fuse


def export2caffe(weights, num_classes, img_size):
    model = MobileNetV2(num_classes)
    weights = torch.load(weights, map_location='cpu')
    model.load_state_dict(weights['model'])
    model.eval()
    fuse(model)
    dummy_input = torch.ones([1, 3, img_size[1], img_size[0]])
    torch.onnx.export(model, dummy_input, 'MobileNetV2.onnx', input_names=['input'], output_names=['output'], opset_version=7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str)
    parser.add_argument('-nc', '--num-classes', type=int, default=21)
    parser.add_argument('-s', '--img_size', type=int, nargs=2, default=[224, 224])
    opt = parser.parse_args()

    export2caffe(opt.weights, opt.num_classes, opt.img_size)
