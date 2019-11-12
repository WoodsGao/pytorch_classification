import numpy as np
import cv2
import torch
import torch.nn as nn

CE = nn.CrossEntropyLoss(reduction='none')
BCE = nn.BCELoss(reduction='none')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_loss(outputs, targets):
    cls_loss = CE(outputs, targets)
    cls_loss = cls_loss.mean()
    loss = cls_loss
    return loss


def show_batch(save_path, inputs, targets, classes):
    imgs = inputs.clone()[:8]
    labels = targets.clone()[:8]
    imgs *= 255.
    imgs = imgs.clamp(0, 255).permute(0, 2, 3, 1).byte().numpy()[:, :, :, ::-1]
    out_imgs = []
    for bi, (img, c) in enumerate(zip(imgs, labels)):
        img = cv2.resize(img, (128, 128))
        cv2.putText(img, classes[c.item()], (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        out_imgs.append(img)

    imgs = np.concatenate(out_imgs, 0)
    save_img = imgs
    cv2.imwrite(save_path, save_img)
