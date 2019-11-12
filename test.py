import torch
from models import GCM
from torch.utils.data import DataLoader
from utils.modules.datasets import ClassificationDataset
from utils.utils import compute_loss, show_batch, device
from tqdm import tqdm
import argparse


def test(model, val_loader):
    model.eval()
    val_loss = 0
    classes = val_loader.dataset.classes
    num_classes = len(classes)
    total_size = 0
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for idx, (inputs, targets) in pbar:
            batch_idx = idx + 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            val_loss += loss.item()
            predicted = outputs.max(1)[1]
            if idx == 0:
                show_batch('test_batch.png', inputs.cpu(), predicted.cpu(),
                           classes)
            eq = predicted.eq(targets)
            total_size += predicted.size(0)
            for c_i, c in enumerate(classes):
                indices = targets.eq(c_i)
                positive = indices.sum().item()
                tpi = eq[indices].sum().item()
                fni = positive - tpi
                fpi = predicted.eq(c_i).sum().item() - tpi
                tp[c_i] += tpi
                fn[c_i] += fni
                fp[c_i] += fpi
                T = tp + fn
                P = tp + fp
                P[P <= 0] = 1
                P = tp / P
                R = tp + fn
                R[R <= 0] = 1
                R = tp / R
                F1 = (2 * tp + fp + fn)
                F1[F1 <= 0] = 1
                F1 = 2 * tp / F1

            pbar.set_description('loss: %8lf, prec: %8lf, F1: %8lf' %
                                 (val_loss / batch_idx, P.mean(), F1.mean()))

    for c_i, c in enumerate(classes):
        print('cls: %8s, targets: %8d, pre: %8lf, rec: %8lf, F1: %8lf' %
              (c, T[c_i], P[c_i], R[c_i], F1[c_i]))
    val_loss /= len(val_loader)
    return val_loss, P.mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-list', type=str, default='data/lsr/valid.txt')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=0)
    opt = parser.parse_args()

    val_data = ClassificationDataset(
        opt.val_list,
        img_size=opt.img_size
    )
    val_loader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    model = GCM(1024)
    model = model.to(device)
    if opt.weights:
        state_dict = torch.load(opt.weights, map_location=device)
        model.load_state_dict(state_dict['model'])
    val_loss, prec = test(model, val_loader)
    print('val_loss: %8g   prec: %8g' % (val_loss, prec))
