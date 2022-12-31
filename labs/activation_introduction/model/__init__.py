import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch import nn, Tensor


class LinearModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 1)
        )
        
        self.loss = nn.MSELoss()
        
    def forward(self, X):
        return self.model(X)
    
    def training_step(self, batch, batch_idx) -> Tensor:
        X, y = batch
        y_hat = self.model(X)
        l = self.loss(y_hat, y)
        self.log("train_loss", l.item())
        
    
        
        
        
            


