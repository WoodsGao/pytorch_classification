import cv2
import numpy as np
import torch
from imgaug import augmenters as ia

from pytorch_modules.utils import device


@torch.no_grad()
def inference(model, img, img_size=(64, 64), rect=False):
    img = img[:, :, ::-1]
    h, w, c = img.shape

    if rect:
        scale = min(img_size[0] / w, img_size[1] / h)
        resize = ia.Sequential([
            ia.Resize({
                'width': int(w * scale),
                'height': int(h * scale)
            }),
            ia.PadToFixedSize(*img_size,
                              position='center')
        ])
    else:
        resize = ia.Resize({'width': img_size[0], 'height': img_size[1]})
    img = resize.augment_image(img)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    imgs = torch.FloatTensor([img]).to(device)
    imgs -= torch.FloatTensor([123.675, 116.28,
                               103.53]).reshape(1, 3, 1, 1).to(imgs.device)
    imgs /= torch.FloatTensor([58.395, 57.12,
                               57.375]).reshape(1, 3, 1, 1).to(imgs.device)
    preds = model(imgs)[0].softmax(0).cpu().numpy()
    return preds
