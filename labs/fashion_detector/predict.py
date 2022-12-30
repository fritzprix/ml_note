
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

FASHION_MNIST_CLASSES = 10
FASHION_MNIST_IMAGE_SIZE = 28

def draw_confusion_matrix(axis: plt.Axes,confusion_matrix: torch.Tensor):
    axis.imshow(confusion_matrix, cmap='Blues')
    axis.set_xticks(range(confusion_matrix.shape[0]))
    axis.set_yticks(range(confusion_matrix.shape[1]))
    axis.set_label("Predicted label")
    # axis.ylabel('True label')


def draw_confusion_matrix_with_predictions(confusion_matrix: torch.Tensor, test_dataset, model):
    # Draw the confusion matrix
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(confusion_matrix.shape[0]))
    plt.yticks(range(confusion_matrix.shape[1]))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Choose random images from the test dataset
    num_images = 9
    random_indices = random.sample(range(len(test_dataset)), num_images)
    random_images = [test_dataset[i][0] for i in random_indices]

    # Predict the labels for the random images
    random_predictions = model(random_images)
    random_predictions = [pred.argmax().item() for pred in random_predictions]

    # Display the random images with their predictions
    plt.subplot(2,3,4)
    for i in range(num_images):
        plt.subplot(2,3,i+4)
        plt.imshow(random_images[i].permute(1,2,0))
        plt.title(f'Prediction: {random_predictions[i]}')
        plt.axis('off')
    plt.show()
    
def draw_confusion_matrix_with_predictions(confusion_matrix: torch.Tensor, pred: torch.Tensor, 
                                           labels: list, dataset, sampler_size: tuple):
    # Draw the confusion matrix
    plt.figure(figsize=(5,5))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(confusion_matrix.shape[0]))
    plt.yticks(range(confusion_matrix.shape[1]))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Convert the logits predictions to class labels
    predictions = pred.argmax(dim=1)

    # Display the random images with their predictions and labels
    _, axs = plt.subplots(nrows=sampler_size[0], ncols=sampler_size[1])
    for row in range(sampler_size[0]):
        for col in range(sampler_size[1]):
            idx = sampler_size[1] * row + col
            axs[row][col].imshow(dataset[idx][0])
            pred = predictions[idx].item()
            label = labels[idx]
            axs[row][col].set_title(f"{dataset.classes[pred]}\n({dataset.classes[label]})", color = 'red' if label is not pred else 'black')
    
    plt.subplots_adjust(hspace=1.5, wspace=1.5)
    plt.show()
    
class ClassifierPipeline(pl.LightningModule):
    
    def __init__(self, model: pl.LightningModule) -> None:
        super().__init__()
        self.model = model
        self.taret_device = model.device
        
    def forward(self, X: list):
        X = torch.cat([Preprocess(x).unsqueeze(0) for x in X])
        return self.model(X)
    
def load_bm_model(model_name, ckpt):
    if model_name == 'resnet':
        model = ResNetBasedClassifier(FASHION_MNIST_IMAGE_SIZE, FASHION_MNIST_CLASSES)
        if args.ckpt is None:
            with wandb.init() as run:
                artifact = run.use_artifact('dwidlee/resnet_fashion_classifier/model-qmj4h546:v28', type='model')
                artifact_dir = artifact.download()
                model = model.load_from_checkpoint(f"{artifact_dir}/model.ckpt")
    elif model_name == 'simple':
        model = SimpleFashionMNISTClassifier(num_classes=FASHION_MNIST_CLASSES)
        if args.ckpt is None:
            with wandb.init() as run:
                artifact = run.use_artifact('dwidlee/minimal_fashion_mnist_classifier/model-qgtrsjr5:v25', type='model')
                artifact_dir = artifact.download()
                model = model.load_from_checkpoint(f"{artifact_dir}/model.ckpt")
    else:
        print(f"invalid model name : {model_name} should be either \"simple\" or \"resnet\"")
        exit(1)
        
    if ckpt is not None:
        model = model.load_from_checkpoint(ckpt)
    
    return model

def benchmark(model_name: str, batch_size=128, device=torch.device('cpu'), sampler_size=(4, 4)):
    
    print(f"start benchmark.. batch_size={batch_size} on [{device}]")
    # load dataset
    test_raw = datasets.FashionMNIST('./data', train=False, download=True)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=Preprocess, download=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count())
    
    num_of_classes = len(test_dataset.classes)
    
    model = load_bm_model(args.model, args.ckpt).to(device)
    model.eval()
    
    trainer = pl.Trainer(accelerator='gpu')
    print(trainer.test(model, test_dataloader))
    
    # draw_confusion_matrix(axs[0], model.cmat)
    pipeline = ClassifierPipeline(model)
    smpls = random.sample(range(0, len(test_raw)), sampler_size[0] * sampler_size[1])
    batch_inputs = [test_raw[i][0] for i in smpls]
    batch_labels = [test_raw[i][1] for i in smpls]
    predicts = pipeline(batch_inputs)
    draw_confusion_matrix_with_predictions(model.cmat, predicts, batch_labels, test_raw, sampler_size)
    
    
    
    
    
def main(args: argparse.Namespace):
    if args.bm:
        cuda = torch.device('cuda')
        benchmark(model_name=args.model, device=cuda)
        exit(0)
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict.py')
    parser.add_argument('--model', default='simple', help="model name should be either \"resnet\" or \"simple\"(default)")
    parser.add_argument('--bm', type=bool, help='benchmark with FashionMNISTtest dataset', default=True)
    parser.add_argument('--bs', type=int, help='batch size', default=128)
    parser.add_argument('--ckpt', default=None, help="model checkpoint file")
    args = parser.parse_args()
    print(f"{args}")
    main(args)
    