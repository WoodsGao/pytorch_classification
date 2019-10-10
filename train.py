from cv_utils.dataloader import ClassifyDataloader
from cv_utils import augments
import torch
from model import SENet
import os
from utils import device
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from test import test
from loss import FocalBCELoss
from torchsummary import summary

print(device)


def train(data_dir,
          epochs=100,
          img_size=224,
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          resume=False,
          resume_path='',
          augments_list=[]):
    if not os.path.exists('weights'):
        os.mkdir('weights')

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_loader = ClassifyDataloader(
        train_dir,
        img_size=img_size,
        batch_size=batch_size,
        augments=augments_list + [
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ],
    )
    val_loader = ClassifyDataloader(
        val_dir,
        img_size=img_size,
        batch_size=batch_size,
        augments=[
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ],
    )
    best_acc = 0
    best_loss = 1000
    epoch = 0
    num_classes = len(train_loader.classes)
    model = SENet(3, num_classes)
    model = model.to(device)
    summary(model, (3, img_size, img_size))
    if resume:
        state_dict = torch.load(resume_path, map_location=device)
        best_acc = state_dict['acc']
        best_loss = state_dict['loss']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'])
    criterion = FocalBCELoss()
    # optimizer = optim.Adam([{
    #     'params': model.backbone.parameters(),
    #     'lr': 1e-2 * lr
    # }, {
    #     'params': model.fc.parameters(),
    #     'weight_decay': 1e-3
    # }],
    #    lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # create dataset

    while epoch < epochs:
        # train
        model.train()
        total_loss = 0
        pbar = tqdm(range(train_loader.iter_times))
        optimizer.zero_grad()
        for batch_idx in pbar:

            inputs, targets = train_loader.next()
            inputs = torch.FloatTensor(inputs).to(device)
            targets = torch.FloatTensor(targets).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            total_loss += loss.item()
            pbar.set_description('train loss: %lf' % (total_loss /
                                                      (batch_idx + 1)))
            if batch_idx % accumulate == accumulate - 1 or \
                    batch_idx + 1 == train_loader.iter_times:
                optimizer.step()
                optimizer.zero_grad()
        # validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        total_c = torch.zeros(num_classes)
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        tn = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)
        with torch.no_grad():
            pbar = tqdm(range(val_loader.iter_times))

            for batch_idx in pbar:
                inputs, targets = val_loader.next()
                inputs = torch.FloatTensor(inputs).to(device)
                targets = torch.FloatTensor(targets).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                predicted = outputs.max(1)[1]
                targets = targets.max(1)[1]
                eq = predicted.eq(targets)
                total += targets.size(0)
                correct += eq.sum().item()
                acc = 100. * correct / total
                loss = val_loss / (batch_idx + 1)

                for c_i, c in enumerate(val_loader.classes):
                    indices = targets.eq(c_i).nonzero()
                    total_c[c_i] += targets.eq(c_i).sum().item()
                    tp[c_i] += eq[indices].sum().item()
                    fn[c_i] += targets.eq(c_i).sum().item() - \
                        eq[indices].sum().item()
                    indices = predicted.eq(c_i).nonzero()
                    tn[c_i] += eq[indices].sum().item()
                    fp[c_i] += predicted.eq(c_i).sum().item() - \
                        eq[indices].sum().item()

                pbar.set_description('loss: %10lf, acc: %10lf' % (loss, acc))

        for c_i, c in enumerate(val_loader.classes):
            print('cls: %10s, targets: %10d, pre: %10lf, rec: %10lf' %
                  (c, total_c[c_i], tp[c_i] / (tp[c_i] + fp[c_i]), tp[c_i] /
                   (tp[c_i] + fn[c_i])))

        # Save checkpoint.
        state_dict = {
            'model': model.state_dict(),
            'acc': acc,
            'loss': loss,
            'epoch': epoch
        }
        torch.save(state_dict, 'weights/last.pth')
        if loss < best_loss:
            print('\nSaving..')
            torch.save(state_dict, 'weights/best_loss.pth')
            best_loss = loss
        elif acc > best_acc:
            print('\nSaving..')
            torch.save(state_dict, 'weights/best_acc.pth')
            best_acc = acc
        if epoch % 10 == 0 and epoch > 1:
            print('\nSaving..')
            torch.save(state_dict, 'weights/backup%d.pth' % epoch)
        epoch += 1


if __name__ == "__main__":
    augments_list = [
        augments.PerspectiveProject(0.4, 0.4),
        augments.HSV_H(0.1, 0.4),
        augments.HSV_S(0.1, 0.4),
        augments.HSV_V(0.1, 0.4),
        augments.Rotate(1, 0.4),
        augments.Blur(0.1, 0.3),
        augments.Noise(0.05, 0.3),
    ]
    data_dir = 'data/road_mark'
    train(data_dir, img_size=64, batch_size=64, augments_list=augments_list)
