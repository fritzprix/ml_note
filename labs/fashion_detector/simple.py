import argparse
import pytorch_lightning as pl
from torch import nn, optim
import torch
import torchmetrics
from tqdm import tqdm
import torchmetrics
import os
from torchvision import datasets, transforms
from torch.utils import data

Preprocess = transforms.Compose([
    transforms.PILToTensor(),
    lambda n: n.float(),
    transforms.Normalize((0.0), (0.5))
])

class SimpleFashionMNISTClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1), nn.ReLU(), # (n,1,28,28) => (n, 8, 27, 27)
            nn.Conv2d(16, 32, kernel_size=3, stride=1), nn.ReLU(), # (n,8,27,27) => (n, 16, 25, 25)
            nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(), 
            nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
        
        self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        
    def forward(self, input):
        return self.network(input)
    
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = nn.functional.cross_entropy(self.network(X), y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.network(X)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.metric(y_hat.argmax(dim=1), y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(),lr=self.lr)
    

def benchmark(ckpt: str, batch_size=128, device=None):
    print(f"start benchmark.. ckpt={ckpt} batch_size={batch_size}")
    # load dataset
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=Preprocess)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count())
    
    num_of_classes = len(test_dataset.classes)
    
    model = SimpleFashionMNISTClassifier(num_classes=num_of_classes)
    model = model.load_from_checkpoint(ckpt)
    model.eval()
    
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_of_classes)
    acc_v = []
    
    for X, y in tqdm(iter(test_dataloader)):
        y_hat = model(X)
        assert(isinstance(y_hat, torch.Tensor))
        accuarcy = accuracy(y_hat.argmax(dim=1), y)
        assert(isinstance(accuarcy, torch.Tensor))
        acc_v.append(accuarcy.item())
    acc_v = torch.Tensor(acc_v)
    
    print(f"Avg. accuracy : {acc_v.mean()} / Worst : {acc_v.min()} / Best : {acc_v.max()}")
    
    
    
    
    
def main(args: argparse.Namespace):
    ckpt = args.ckpt
    if args.bm:
        benchmark(ckpt=ckpt)
        exit(0)
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('simple Fashion MNIST classifier')
    parser.add_argument('--ckpt', default='./models/epoch=29-step=12660-v1.ckpt')
    parser.add_argument('--bm', type=bool, help='benchmark with FashionMNISTtest dataset', default=True)
    parser.add_argument('--bs', type=int, help='batch size', default=128)
    args = parser.parse_args()
    print(f"{args}")
    main(args)
    