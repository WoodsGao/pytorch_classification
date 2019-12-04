import torch
from models import EfficientNetGCM
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.modules.datasets import ClassificationDataset
from utils.modules.utils import device
from utils.utils import compute_loss, show_batch, compute_metrics
from tqdm import tqdm
import argparse


def test(model, fetcher):
    model.eval()
    val_loss = 0
    classes = fetcher.loader.dataset.classes
    num_classes = len(classes)
    total_size = torch.Tensor(0)
    true_size = torch.Tensor(0)
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    with torch.no_grad():
        pbar = tqdm(enumerate(fetcher), total=len(fetcher))
        for idx, (inputs, targets) in pbar:
            batch_idx = idx + 1
            outputs = model(inputs)
            loss = compute_loss(outputs, targets, model)
            val_loss += loss.item()
            predicted = outputs.max(1)[1]
            if idx == 0:
                show_batch('test_batch.png', inputs.cpu(), predicted.cpu(),
                           classes)
            eq = predicted.eq(targets)
            total_size += predicted.size(0)
            true_size += eq.size()
            for c_i, c in enumerate(classes):
                indices = targets.eq(c_i)
                positive = indices.sum().item()
                tpi = eq[indices].sum().item()
                fni = positive - tpi
                fpi = predicted.eq(c_i).sum().item() - tpi
                tp[c_i] += tpi
                fn[c_i] += fni
                fp[c_i] += fpi
            pbar.set_description('loss: %8g, acc: %8g' %
                                 (val_loss / batch_idx, true_size / total_size))
    if dist.is_initialized():
        tp = tp.to(device)
        fn = fn.to(device)
        fp = fp.to(device)
        total_size = total_size.to(device)
        true_size = true_size.to(device)
        dist.all_reduce(tp, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(true_size, op=dist.ReduceOp.SUM)
        T, P, R, miou, F1 = compute_metrics(tp.cpu(), fn.cpu(), fp.cpu())
    for c_i, c in enumerate(classes):
        print('cls: %8s, targets: %8d, pre: %8g, rec: %8g, F1: %8g' %
              (c, T[c_i], P[c_i], R[c_i], F1[c_i]))
    return true_size / total_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-list', type=str, default='data/lsr/valid.txt')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--num-workers', type=int, default=0)
    opt = parser.parse_args()

    val_data = ClassificationDataset(opt.val_list, img_size=opt.img_size)
    val_loader = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    model = EfficientNetGCM(20)
    model = model.to(device)
    if opt.weights:
        state_dict = torch.load(opt.weights, map_location=device)
        model.load_state_dict(state_dict['model'])
    val_loss, prec = test(model, val_loader)
    print('val_loss: %8g   prec: %8g' % (val_loss, prec))
