import os
import os.path as osp
import argparse
from tqdm import tqdm
import torch
from models import ResNet18
from pytorch_modules.utils import device, IMG_EXT
from utils.inference import inference
import cv2


def run(img_dir, outputs, weights, img_size, num_classes, rect):
    results = []
    model = ResNet18(num_classes)
    state_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    names = [n for n in os.listdir(img_dir) if osp.splitext(n)[1] in IMG_EXT]
    for name in tqdm(names):
        path = osp.join(img_dir, name)
        img = cv2.imread(path)
        pred = inference(model, img, img_size, rect=rect)
        idx = pred.argmax()
        results.append('%s %d' % (path, idx))
    with open(outputs, 'w') as f:
        f.write('\n'.join(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs', type=str, default='data/samples')
    parser.add_argument('outputs', type=str, default='outputs.txt')
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    parser.add_argument('-s',
                        '--img_size',
                        type=int,
                        nargs=2,
                        default=[224, 224])
    parser.add_argument('-nc', '--num-classes', type=int, default=10)
    parser.add_argument('--rect', action='store_true')
    opt = parser.parse_args()

    run(opt.imgs, opt.outputs, opt.weights, opt.img_size, opt.num_classes,
        opt.rect)
