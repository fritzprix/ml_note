import pytorch_lightning as pl
import torch
from torch.nn import functional as F

class GRUML(pl.LightningModule):
    def __init__(self, vocab_size, num_hidden=10, num_layers=2):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.net = torch.nn.GRU(input_size=vocab_size, 
                                hidden_size=num_hidden, 
                                num_layers=num_layers)
        
    def forward(self,X):
        embedding = F.one_hot(X, self.vocab_size)
        y_hat = self.net(embedding)
        
        
        
        
        
