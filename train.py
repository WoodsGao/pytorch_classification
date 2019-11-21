import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import GCM
import os
from utils.utils import device, compute_loss, show_batch
from utils.modules.datasets import ClassificationDataset
from tqdm import tqdm
from test import test
# from torchsummary import summary
import random
import argparse

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

print(device)
writer = SummaryWriter()
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True

augments = {
    'hsv': 0.1,
    'blur': 0.1,
    'pepper': 0.1,
    'shear': 0.1,
    'translate': 0.1,
    'rotate': 0.1,
    'flip': 0.1,
    'scale': 0.1,
    'noise': 0.1,
}


def train(data_dir,
          epochs=100,
          img_size=224,
          batch_size=8,
          accumulate=2,
          lr=1e-3,
          resume=False,
          weights='',
          num_workers=0,
          multi_scale=False,
          adam=False,
          no_test=False):
    os.makedirs('weights', exist_ok=True)
    if multi_scale:
        img_size_min = max(img_size * 0.67 // 32, 1)
        img_size_max = max(img_size * 1.5 // 32, 1)
    train_list = os.path.join(data_dir, 'train.txt')
    val_list = os.path.join(data_dir, 'valid.txt')
    train_data = ClassificationDataset(train_list,
                                       img_size=img_size,
                                       augments=augments)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_data = ClassificationDataset(
        val_list,
        img_size=img_size,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    accumulate_count = 0
    best_prec = 0
    best_loss = 1000
    epoch = 0
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    model = GCM(num_classes)
    model = model.to(device)
    # optimizer = AdaBoundW(model.parameters(), lr=lr, weight_decay=5e-4)
    if adam:
        optimizer = optim.AdamW(model.parameters(),
                                lr=lr if lr > 0 else 1e-4,
                                weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=lr if lr > 0 else 1e-3,
                              momentum=0.9,
                              weight_decay=1e-5,
                              nesterov=True)
    if resume:
        state_dict = torch.load(weights, map_location=device)
        if adam:
            if 'adam' in state_dict:
                optimizer.load_state_dict(state_dict['adam'])
        else:
            if 'sgd' in state_dict:
                optimizer.load_state_dict(state_dict['sgd'])
        if lr > 0:
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        best_prec = state_dict['prec']
        best_loss = state_dict['loss']
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'], strict=False)

    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    # scheduler = lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[round(epochs * x) for x in [0.8, 0.9]],
    #     gamma=0.1,
    # )
    # scheduler.last_epoch = epoch - 1

    # summary(model, (3, img_size, img_size))

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level='O1',
                                          verbosity=0)
    # create dataset
    while epoch < epochs:
        print('%d/%d' % (epoch, epochs))
        # train
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()
        for idx, (inputs, targets) in pbar:
            batch_idx = idx + 1
            if idx == 0:
                show_batch('train_batch.png', inputs, targets, classes)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if multi_scale:
                img_size = random.randrange(img_size_min, img_size_max) * 32
            if multi_scale:
                inputs = F.interpolate(inputs,
                                       size=img_size,
                                       mode='bilinear',
                                       align_corners=False)
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            total_loss += loss.item()
            loss /= accumulate
            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            accumulate_count += 1
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available(
            ) else 0  # (GB)
            pbar.set_description('train mem: %5.2lfGB loss: %8lf scale: %8d' %
                                 (mem, total_loss / batch_idx, inputs.size(2)))
            if accumulate_count % accumulate == 0:
                accumulate_count = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                optimizer.zero_grad()
        torch.cuda.empty_cache()
        writer.add_scalar('train_loss', total_loss / len(train_loader), epoch)
        print('')
        # validate
        val_loss = best_loss
        prec = best_prec
        if not no_test:
            val_loss, prec = test(model, val_loader)
            writer.add_scalar('valid_loss', val_loss, epoch)
            writer.add_scalar('prec', prec, epoch)
        epoch += 1
        for pg in optimizer.param_groups:
            pg['lr'] *= (1 - 1e-8)
        # Save checkpoint.
        state_dict = {
            'model': model.state_dict(),
            'prec': prec,
            'loss': val_loss,
            'epoch': epoch
        }
        if adam:
            state_dict['adam'] = optimizer.state_dict()
        else:
            state_dict['sgd'] = optimizer.state_dict()
        torch.save(state_dict, 'weights/last.pt')
        if val_loss < best_loss:
            print('\nSaving best_loss.pt..')
            torch.save(state_dict, 'weights/best_loss.pt')
            best_loss = val_loss
        if prec > best_prec:
            print('\nSaving best_prec.pt..')
            torch.save(state_dict, 'weights/best_prec.pt')
            best_prec = prec
        if epoch % 10 == 0 and epoch > 1:
            print('\nSaving backup%d.pt..' % epoch)
            torch.save(state_dict, 'weights/backup%d.pt' % epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/road_mark')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--weights', type=str, default='weights/last.pt')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--no-test', action='store_true')
    opt = parser.parse_args()
    train(
        data_dir=opt.data_dir,
        epochs=opt.epochs,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        lr=opt.lr,
        resume=opt.resume,
        weights=opt.weights,
        adam=opt.adam,
        num_workers=opt.num_workers,
        multi_scale=opt.multi_scale,
        no_test=opt.no_test,
    )
