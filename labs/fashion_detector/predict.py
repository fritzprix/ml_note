import argparse
import random
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from torch import nn, optim
import torch
import torchmetrics
from tqdm import tqdm
import torchmetrics
import os
import wandb
from torchvision import datasets, transforms
from torch.utils import data
from models import SimpleFashionMNISTClassifier, Preprocess, ResNetBasedClassifier
from pytorch_lightning import loggers

    
class ClassifierPipeline(pl.LightningModule):
    
    def __init__(self, model: pl.LightningModule) -> None:
        super().__init__()
        self.model = model
        self.taret_device = model.device
        
    def forward(self, X: list):
        X = torch.cat([Preprocess(x).unsqueeze(0) for x in X])
        return self.model(X)

def benchmark(model_name: str, batch_size=128, device=torch.device('cpu'), sampler_size=(4, 4)):
    
    print(f"start benchmark.. batch_size={batch_size} on [{device}]")
    # load dataset
    test_raw = datasets.FashionMNIST('./data', train=False, download=True)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=Preprocess, download=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count())
    
    num_of_classes = len(test_dataset.classes)
    
    if model_name == 'resnet':
        input_size = next(iter(test_dataset))[0].shape[2]
        model = ResNetBasedClassifier(input_size, num_of_classes)
        with wandb.init() as run:
            artifact = run.use_artifact('dwidlee/resnet_fashion_classifier/model-qmj4h546:v28', type='model')
            artifact_dir = artifact.download()
            model = model.load_from_checkpoint(f"{artifact_dir}/model.ckpt")
    elif model_name == 'simple':
        model = SimpleFashionMNISTClassifier(num_classes=num_of_classes)
        with wandb.init() as run:
            artifact = run.use_artifact('dwidlee/minimal_fashion_mnist_classifier/model-qgtrsjr5:v25', type='model')
            artifact_dir = artifact.download()
            model = model.load_from_checkpoint(f"{artifact_dir}/model.ckpt")
    else:
        print(f"invalid model name : {model_name} should be either \"simple\" or \"resnet\"")
        exit(1)
        
    model.to(device)
    model.eval()
    
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_of_classes)
    accuracy.to(device)
    
    
    trainer = pl.Trainer(accelerator='gpu')
    print(trainer.test(model, test_dataloader))
    
    
    pipeline = ClassifierPipeline(model)
    _, axs = plt.subplots(nrows=sampler_size[0], ncols=sampler_size[1])
    smpls = random.sample(range(0, len(test_raw)), sampler_size[0] * sampler_size[1])
    batch_inputs = [test_raw[i][0] for i in smpls]
    batch_labels = [test_raw[i][1] for i in smpls]
    predicts = pipeline(batch_inputs).argmax(dim=1)
    for row in range(sampler_size[0]):
        for col in range(sampler_size[1]):
            idx = sampler_size[1] * row + col
            axs[row][col].imshow(batch_inputs[idx])
            pred = predicts[idx].item()
            label = batch_labels[idx]
            axs[row][col].set_title(f"{test_raw.classes[pred]}\n({test_raw.classes[label]})", color = 'red' if label is not pred else 'black')
    
    plt.subplots_adjust(hspace=1.5, wspace=1.5)
    plt.show()
    
    
    
def main(args: argparse.Namespace):
    if args.bm:
        cuda = torch.device('cuda')
        benchmark(model_name=args.model, device=cuda)
        exit(0)
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('simple Fashion MNIST classifier')
    parser.add_argument('--model', default='simple')
    parser.add_argument('--bm', type=bool, help='benchmark with FashionMNISTtest dataset', default=True)
    parser.add_argument('--bs', type=int, help='batch size', default=128)
    parser.add_argument('--ckpt', default=None)
    args = parser.parse_args()
    print(f"{args}")
    main(args)
    