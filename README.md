# Changelog

## Optimizations and Enhancements

- **Dataset Caching**  
  - **Issue Identified:** Loading the dataset from disk (HDD/SSD) was causing a significant bottleneck.  
    - **Before:** During training CPU usage was at 100% and GPU usage hovered between 0–60% due to the overhead of reading and preprocessing images on-the-fly.  
    - **After:** By preloading the dataset into RAM, the bottleneck was eliminated. This resulted in a dramatic reduction of CPU usage (now around 20%) and improved GPU utilization (now over 80%).
  - Preloaded images and annotations into memory, thus removing repeated disk I/O operations during training.
  - This reduced training time by over 6 times.

- **DataLoader Improvements**  
  - Adjusted DataLoader settings such as `pin_memory` and `prefetch_factor` to further streamline data throughput.

## Primary Bottleneck Addressed

The main performance issue was due to the slow disk-based loading of the dataset. By caching the dataset in memory, we significantly reduced the CPU overhead and allowed the GPU to run at higher utilization, leading to overall faster training times.





PyTorch implementation of:

* [SSDLite: Lightweight SSD based on MobileNet](https://arxiv.org/abs/1801.04381) 

To make it friendly to HW based NPU, 

* variance for anchor's xy and wh is changed from `0.1, 0.2` to `0.125, 0.125`
* L2Norm layer is removed.
* atrous convolution is replaced to normal convolution.

# Evaluation Results on VOC2007 test dataset

| model         | augumentation | training set          | epochs | loss  | mAP    |
|:-------------:|:-------------:|:---------------------:|:------:|:-----:|:------:|
| ssdlite320+rfb| O             | VOC2007/2012 trainval | 200    | 2.296 | 0.741  |


optimizer: ADMAW scheduler: cosineLR

# Pre-requisite

```
pip install -r requirements.txt
```

# Train

train with default parameter (dataset will be downloaded automatically)

```
python detect_train.py
```

## Command Arguments
| name                     | description                              | default   |
|--------------------------|------------------------------------------|:---------:|
| `--model`                | model name                               | ssdlite320|
| `--dataset_root`         | dataset location                         |           |
| `--dataset_domains`      | dataset(train, validation, test)         |           |
| `--epochs`               | number of epochs to run                  | 200       |
| `--batch_size`           | size of mini-batch                       | 32        |
| `--lr`                   | learning rate for SGD                    | 1e-3      |
| `--num_workers`          | Number of workers used in dataloading    | -1        |
| `--weight_decay`         | weight decay for SGD                     | 5e-4      |
| `--gamma`                | gamma for lr scheduler                   | 0.1       |
| `--th_conf`              | confidence threshold                     | 0.5       |
| `--th_iou`               | iou threshold                            | 0.5       |
| `--resume`               | resume training                          | None      |
| `--momentum`             | Momentum value for optimizer             | 0.9       |
| `--step_size`            | `step_size` for step lr scheduler        | 30        |
| `--scheduler`            | use scheduler                            | multi_step|
| `--optimizer`            | use optimizer                            | sgd       |
| `--milestones`           | `milestones` for multi step lr scheduler | 140 170   |
| `--disable_augmentation` | disable random augmentation              | False     |
| `--random_expand`        | usde the dataset is mainly in a large box| True      |
| `--enable_letterbox`     | enable letter boxing image               | False     |


## Available models
| name                    | input_size | backbone    | target |
|:-----------------------:|:----------:|:-----------:|:------:|
| ssdlite320              | 320x320    | MobileNetV2 | VOC    |
| ssdlite512              | 512x512    | MobileNetV2 | VOC    |


## Example
Training SSDlite with multi step lr and batch_size is 32 (default)


Resume training
```
python detect_train.py --resume checkpoints/ssdlite300_latest.pth
```

# Evaluation
calculate mAP with test image set

```
python detect_eval.py
```

## Command Arguments
| name                  | description       | default   |
|-----------------------|-------------------|:---------:|
| `--model`             | model name        | ssdlite320|
| `--dataset_domains`   | dataset(test)     | test      |
| `--dataset_root`      | dataset location  |           |
| `--class_path`        | class label txt   | PASCAL_VOC.txt |
| `--weight`            | weight file name  | `checkpoints/{MODEL_NAME}_latest.pth` |
| `--enable_letterbox`  | enable letter boxing image | False |
| `--th_conf`           | confidence threshold                     | 0.05    |
| `--th_iou`            | iou threshold                            | 0.5     |
| `--num_workers`       | Number of workers used in dataloading    | -1      |

# Single run
```
python detect_single.py image1 [image2] [image3] [...]
```

## Command Arguments
| name                  | description                 | default   |
|-----------------------|-----------------------------|:-------:  |
| `--model`             | model name                  | ssdlite320|
| `--weight`            | weight file name            | `checkpoints/{MODEL_NAME}_latest.pth` |
| `--th_conf`           | confidence threshold        | 0.5       |
| `--th_iou`            | iou threshold               | 0.5       |
| `--enable_letterbox`  | enable letter boxing image  | False     |
| `--class_path`        | Class label txt             | PASCAL_VOC.txt|
| `--export`            | export network to onnx file | False     |

