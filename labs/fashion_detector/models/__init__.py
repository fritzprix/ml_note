import pytorch_lightning as pl
from torch import nn, optim
import torch
import torchmetrics
import torchmetrics
from torchvision import transforms, models

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
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.network(X)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.metric(y_hat.argmax(dim=1), y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
    
    
class ResNetBasedClassifier(pl.LightningModule):
    def __init__(self, input_image_size, num_class, lr=1e-5, is_base_frozen=True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if is_base_frozen else None)
        if is_base_frozen:
            for param in resnet.parameters():
                param.requires_grad = False
        
        self.network = nn.Sequential(
            transforms.Resize(224),
            nn.Conv2d(1,3, kernel_size=1),
            resnet,
            nn.ReLU(),
            nn.Linear(resnet.fc.out_features, num_class)
        )
        
        self.loss = nn.CrossEntropyLoss()
        self.metric = torchmetrics.Accuracy('multiclass', num_classes=num_class)
        
    def forward(self, X):
        return self.network(X)
    
    def training_step(self, batch, batch_index) -> torch.Tensor:
        X, y = batch
        loss = self.loss(self.network(X), y)
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, batch_index):
        X, y = batch
        y_hat = self.network(X)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        val_acc = self.metric(y_hat.argmax(dim=1), y)
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.network(X)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        val_acc = self.metric(y_hat.argmax(dim=1), y)
        self.log("test_loss", val_loss)
        self.log("test_acc", val_acc)
    
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.lr)
        
