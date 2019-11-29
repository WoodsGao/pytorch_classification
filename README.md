# pytorch_classification

## Introduction

Implementation of some classification models with pytorch, including SENet, ResNet, DenseNet, etc.

## Features

 - Advanced neural network models
 - Flexible and efficient toolkit(See [woodsgao/pytorch_modules](https://github.com/woodsgao/pytorch_modules))
 - Online data augmenting(See [woodsgao/image_augments](https://github.com/woodsgao/image_augments))
 - Mixed precision training(If you have already installed [apex](https://github.com/NVIDIA/apex))

## Installation

    git clone https://github.com/woodsgao/pytorch_classification
    cd pytorch_classification
    pip install -r requirements.txt

## Usage

### Create custom data

Please organize your data in the following format:

    data/
        <custom>/
            <class_name_1>/
                0001.png
                0002.png
                ...
            <class_name_2>/
                0001.png
                0002.png
                ...

Then execute `python3 split_dataset.py data/<custom>` . It splits the data into training and validation sets and generates `data/<custom>/train.txt` and `data/<custom>/valid.txt` .

### Training

    python3 train.py --data-dir data/<custom> --img-size 224 --batch-size 8 --accumulate 8 --epoch 200 --lr 1e-4 --adam

### Distributed Training

Run the following command in all nodes.The process with rank 0 will save your weights
    python3 -m torch.distributed.launch --nnodes <nnodes> --node_rank <node_rank> --nproc_per_node <nproc_per_node>  --master_addr <master_addr>  --master_port <master_port> train.py --data-dir data/<custom> --img-size 224 --batch-size 8 --accumulate 8 --epoch 200 --lr 1e-4 --adam

### Testing

    python3 test.py --val-list /data/<custom>/valid.txt --img-size 224 --batch-size 8

### Inference

    python3 inference.py --img-dir data/samples --img-size 224 --output_path outputs.csv --weights weights/best_prec.pt
