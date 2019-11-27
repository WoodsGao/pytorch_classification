import os
import argparse
from tqdm import tqdm
import torch
from models import EfficientNet
from utils.utils import device
import cv2


def inference(img_dir='data/samples',
              img_size=224,
              output_path='outputs.csv',
              weights='weights/best_prec.pt'):
    outputs = ''
    model = EfficientNet(20)
    model = model.to(device)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    names = [
        n for n in os.listdir(img_dir)
        if os.path.splitext(n)[1] in ['.jpg', '.jpeg', '.png', '.tiff']
    ]
    with torch.no_grad():
        for name in tqdm(names):
            path = os.path.join(img_dir, name)
            img = cv2.imread(path)
            h = (img.shape[0] / max(img.shape[:2]) * img_size) // 32
            w = (img.shape[1] / max(img.shape[:2]) * img_size) // 32
            img = cv2.resize(img, (int(w * 32), int(h * 32)))
            img = img[:, :, ::-1]
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor([img], device=device) / 255.
            output = model(img).softmax(1).max(1)
            outputs += '%s, %5lf, %d\n' % (path, output[0], output[1])
    with open(output_path, 'w') as f:
        f.write(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='data/samples')
    parser.add_argument('--output-path', type=str, default='outputs.csv')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--weights', type=str, default='weights/best_prec.pt')
    opt = parser.parse_args()
    inference(opt.img_dir, opt.img_size, opt.output_path, opt.weights)
