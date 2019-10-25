import torch
from model import SENet
import os
from utils import device
from utils.dataloader import Dataloader, show_batch
from utils import augments
from utils.loss import FocalBCELoss
from utils.optim import AdaBoundW
from tqdm import tqdm
from test import test
from torchsummary import summary
import argparse

print(device)


def train(data_dir,
          epochs=100,
          img_size=224,
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          resume=False,
          resume_path='',
          augments_list=[],
          multi_scale=False):
    if not os.path.exists('weights'):
        os.mkdir('weights')

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_loader = Dataloader(
        train_dir,
        img_size=img_size,
        batch_size=batch_size,
        augments=augments_list + [
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ],
        multi_scale=multi_scale
    )
    val_loader = Dataloader(
        val_dir,
        img_size=img_size,
        batch_size=batch_size,
        augments=[
            augments.BGR2RGB(),
            augments.Normalize(),
            augments.NHWC2NCHW(),
        ],
    )
    show_batch('train_batch.png', train_loader.data_list[:8], img_size, train_loader.classes, augments_list)
    show_batch('val_batch.png', val_loader.data_list[:8], img_size, val_loader.classes)
    best_acc = 0
    best_loss = 1000
    epoch = 0
    num_classes = len(train_loader.classes)
    model = SENet(num_classes)
    model = model.to(device)
    criterion = FocalBCELoss(alpha=0.25, gamma=2)
    optimizer = AdaBoundW(model.parameters(), lr=lr, weight_decay=5e-4)
    # summary(model, (3, img_size, img_size))
    if resume:
        state_dict = torch.load(resume_path, map_location=device)
        best_acc = state_dict['acc']
        best_loss = state_dict['loss']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'], strict=False)
        optimizer.load_state_dict(state_dict['optimizer'])

    # create dataset
    against_examples = []
    while epoch < epochs:
        print('%d/%d' % (epoch, epochs))
        # train
        model.train()
        total_loss = 0
        pbar = tqdm(range(1, train_loader.iter_times + 1))
        optimizer.zero_grad()
        for batch_idx in pbar:
            inputs, targets = train_loader.next()
            inputs = torch.FloatTensor(inputs).to(device)
            targets = torch.FloatTensor(targets).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            against_examples.append(
                [inputs[loss > 2 * loss.mean()], targets[loss > 2 * loss.mean()]])
            loss.mean().backward()
            total_loss += loss.mean().item()
            pbar.set_description('train loss: %10lf scale: %10d' % (total_loss / batch_idx, inputs.size(2)))
            if batch_idx % accumulate == 0 or \
                    batch_idx == train_loader.iter_times:
                optimizer.step()
                optimizer.zero_grad()
                # against examples training
                for example in against_examples:
                    against_inputs = example[0]
                    if against_inputs.size(0) < 2:
                        continue
                    against_targets = example[1]
                    outputs = model(against_inputs)
                    loss = criterion(outputs, against_targets)
                    loss /= batch_size
                    loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                against_examples = []
        # validate
        val_loss, acc = test(model, val_loader, criterion)
        # Save checkpoint.
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'loss': val_loss,
            'epoch': epoch
        }
        torch.save(state_dict, 'weights/last.pt')
        if val_loss < best_loss:
            print('\nSaving best_loss.pt..')
            torch.save(state_dict, 'weights/best_loss.pt')
            best_loss = val_loss
        if acc > best_acc:
            print('\nSaving best_acc.pt..')
            torch.save(state_dict, 'weights/best_acc.pt')
            best_acc = acc
        if epoch % 10 == 0 and epoch > 1:
            print('\nSaving backup%d.pt..' % epoch)
            torch.save(state_dict, 'weights/backup%d.pt' % epoch)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/lsr')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--multi_scale', action='store_true')
    augments_list = [
        augments.PerspectiveProject(0.3, 0.1),
        augments.HSV_H(0.3, 0.2),
        augments.HSV_S(0.3, 0.2),
        augments.HSV_V(0.3, 0.2),
        augments.Rotate(1, 0.1),
        augments.Blur(0.03, 0.1),
        augments.Noise(0.3, 0.1),
    ]
    opt = parser.parse_args()
    train(data_dir=opt.data_dir,
          epochs=opt.epochs,
          img_size=opt.img_size,
          batch_size=opt.batch_size,
          accumulate=opt.accumulate,
          lr=opt.lr,
          resume=opt.resume,
          resume_path=opt.resume_path,
          augments_list=augments_list,
          multi_scale=opt.multi_scale)
