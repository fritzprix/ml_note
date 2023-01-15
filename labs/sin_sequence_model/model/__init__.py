import pytorch_lightning as pl
import torch
from torch.utils import data
import math
from torch import nn


class GRUAutoregress(pl.LightningModule):
    
    class SelectItem(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            
        def forward(self, inputs) -> torch.Tensor:
            return inputs[self.idx]
            

    def __init__(self, input_size, hidden_size:int = 10, num_layers:int=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, input_size)
        self.loss = nn.MSELoss()
        
        
    def forward(self, X):
        assert(isinstance(X, torch.Tensor))
        gru_out, _ = self.gru(X)
        return self.fc(gru_out)
    
    def predict_k_step(self, X, k_step):
        hout = None
        for _ in range(k_step):
            gru_out, hout = self.gru(X, hout)
            X = self.fc(gru_out)
        return X
            
    
    
    def training_step(self, batch, _) -> torch.Tensor:
        X, y = batch
        return self.loss(self(X), y)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, _):
        X,y = batch
        assert(isinstance(X, torch.Tensor))
        assert(isinstance(y, torch.Tensor))
        y_hat = self(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return {"val_loss": loss}
    
    def test_step(self, batch, _):
        X,y = batch
        assert(isinstance(X, torch.Tensor))
        assert(isinstance(y, torch.Tensor))
        y_hat = self(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        return {"test_loss": loss}



class FFNAutoregressor(pl.LightningModule):
    def __init__(self, lr, n_steps: int, n_features: int, barrel_scale: int, num_hidden: int = 2):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        input_size = n_steps * n_features
        barrel_size = input_size * barrel_scale
        hidden_layers = [nn.Sequential(nn.Linear(barrel_size, barrel_size), nn.ReLU()) for _ in range(num_hidden)]
       
        self.net = nn.Sequential(
            nn.Linear(input_size, barrel_size), nn.ReLU(),
            *hidden_layers,
            nn.Linear(barrel_size, input_size)
        )
        
        self.loss = nn.MSELoss()
        
    def forward(self, X: torch.Tensor):
        # shape of X B,L,F & F is 1
        X = X.squeeze(dim=2)
        # now shape of X  B, L x F
        return self.net(X).unsqueeze(2)
    
    def predict_k_step(self, X, k_step):
        for _ in range(k_step):
            X = self(X)
        return X
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())
    
    def training_step(self, batch, _) -> torch.Tensor:
        X, y = batch
        assert(isinstance(X, torch.Tensor))
        assert(isinstance(y, torch.Tensor))
        y_hat = self(X)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, _):
        X, y = batch
        assert(isinstance(X, torch.Tensor))
        assert(isinstance(y, torch.Tensor))
        y_hat = self(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return {"val_loss": loss}
    
    def test_step(self, batch, _):
        X, y = batch
        assert(isinstance(X, torch.Tensor))
        assert(isinstance(y, torch.Tensor))
        y_hat = self(X)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        return {"test_loss": loss}


class SlicedDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            return SlicedDataset(self.data[index], self.labels[index])
        return self.data[index].unsqueeze(1), self.labels[index].unsqueeze(1)
    
    def __len__(self):
        return len(self.labels)
        
        

class SinusoidalTimeSeries(data.Dataset):
    
    def __init__(self, start, n_cycles:int = 1, n_samples:int = 2000, n_steps:int = 10, label_offset:int = 1, noise:float=0):
        super().__init__()
        start = math.pi * start * 2
        end = math.pi * n_cycles * 2
        sin_w = torch.sin(torch.arange(start, end, (end - start) / float(n_samples))) + torch.normal(mean=0, std=noise, size=(n_samples,))
        self.data = torch.stack([sin_w[i: i + n_steps] for i in range(len(sin_w) - n_steps)])
        self.labels = self.data[label_offset:-1]
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            return SlicedDataset(self.data[index], self.labels[index])
        else:
            return self.data[index].unsqueeze(1), self.labels[index].unsqueeze(1)    
    
    def __len__(self):
        return len(self.labels)
    
        
        
    
        
    