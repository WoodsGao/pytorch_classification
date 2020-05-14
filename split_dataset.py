import argparse
import os
import os.path as osp
import random

from pytorch_modules.utils import IMG_EXT


def run(data_dir, train_rate=0.7, shuffle=True):
    """根据数据文件夹中的images生成所需要的train.txt和valid.txt
    
    Arguments:
        data_dir {str} -- 数据文件夹路径
    
    Keyword Arguments:
        train_rate {float} -- train所占比例 (default: {0.7})
        shuffle {bool} -- 是否打乱顺序 (default: {True})
    """
    class_names = [n for n in os.listdir(data_dir)]
    class_names = [osp.join(data_dir, cn) for cn in class_names]
    class_names = [cn for cn in class_names if osp.isdir(cn)]
    img_names = []
    for cn in class_names:
        names = os.listdir(cn)
        names = [name for name in names if osp.splitext(name)[1] in IMG_EXT]
        names.sort()
        names = [osp.join(osp.basename(cn), name) for name in names]
        img_names += names
    if shuffle:
        random.shuffle(img_names)
    # names = [osp.abspath(name) for name in names]
    with open(osp.join(data_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(img_names[:int(train_rate * len(img_names))]))
    with open(osp.join(data_dir, 'valid.txt'), 'w') as f:
        f.write('\n'.join(img_names[int(train_rate * len(img_names)):]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default='./voc')
    args = parser.parse_args()
    run(args.path)
