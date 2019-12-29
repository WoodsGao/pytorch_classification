import numpy as np
import cv2
import torch
from pytorch_modules.utils import device


@torch.no_grad()
def inference(model, imgs, img_size=(64, 64)):
    imgs = [
        cv2.resize(img, img_size)[:, :, ::-1].transpose(2, 0, 1).astype(
            np.float32) / 255. for img in imgs
    ]
    imgs = torch.FloatTensor(imgs).to(device)
    preds = model(imgs).softmax(1).cpu().numpy().argmax(1).tolist()
    return preds
