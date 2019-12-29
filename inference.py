import os
import os.path as osp
import argparse
from tqdm import tqdm
import torch
from pytorch_modules.backbones.efficientnet import efficientnet
from pytorch_modules.backbones.resnet import resnet18, resnext50_32x4d, resnext101_32x8d
from pytorch_modules.utils import device, IMG_EXT
from utils.inference import inference
import cv2


def run(img_dir='data/samples',
        img_size=(224, 224),
        num_classes=10,
        output_path='outputs.txt',
        weights='weights/best.pt'):
    outputs = []
    model = resnet18(num_classes)
    state_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    names = [n for n in os.listdir(img_dir) if osp.splitext(n)[1] in IMG_EXT]
    for name in tqdm(names):
        path = osp.join(img_dir, name)
        img = cv2.imread(path)
        idx = inference(model, [img], img_size)[0]
        outputs.append('%s %d' % (path, idx))
    with open(output_path, 'w') as f:
        f.write('\n'.join(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='data/samples')
    parser.add_argument('--output-path', type=str, default='outputs.txt')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    opt = parser.parse_args()
    inference(opt.img_dir, opt.img_size, opt.output_path, opt.weights)
