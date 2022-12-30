# CNN model experiment for classification task

## Tools

- ML framework : [pytorch](https://pytorch.org/) & [Lightning](https://github.com/Lightning-AI/lightning)
- Visualization : [wandb](https://wandb.ai/home)

## Performance (Tested on FashionMNIST Test Dataset)

|Name|Avg. Accuracy|
|:-:|:-:|
|simple|91.9%|
|resnet|~|

## Running Environment

- WSL2 @ Windows 11
- [Python environment](./conda.env)

## Usage

```bash
usage: predict.py [-h] [--model MODEL] [--bm BM] [--bs BS] [--ckpt CKPT]

options:
  -h, --help     show this help message and exit
  --model MODEL  model name should be either "resnet" or "simple"(default)
  --bm BM        benchmark with FashionMNISTtest dataset
  --bs BS        batch size
  --ckpt CKPT    model checkpoint file
```

## Try out

> To see benchmark result for simple classifier model

```bash
$ python predict.py
```

> To see BM result for ResNet50 based model
```bash
$ python predict.py
```

## TO-DO

- [x] building a imple CNN model for classification task
- [x] buinding a ResNet50 based classifier for FashionMNIST
