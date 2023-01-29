import pytorch_lightning as pl
import torch
from torch import nn, Tensor, int64


class KorEncoder(pl.LightningModule):
    def __init__(self, input_size:int, embed_size, hidden_size:int, num_layers:int=2, lr:float=1e-4, padding_id=0) -> None:
        super(KorEncoder, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.padding_id = padding_id
        self.embedding = nn.Embedding(input_size, embed_size)
        self.enc = nn.GRU(input_size=embed_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers,
                          bidirectional=True)
        
        self.fc = nn.LazyLinear(input_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_id)
        
    def forward(self, X, state=None):
        # X (N,L,IV) => Torch RNN (L,N,IV)
        assert isinstance(X, Tensor)
        emb_X = self.embedding(X.t().type(int64))
        outputs, state = self.enc(emb_X, state)
        # output will (L,N,2 * hidden_size)
        # state will (2 * num_layers, hidden_size)
        return outputs, state
        
    def training_step(self, batch, _) -> Tensor:
        X, y = batch[:, :-1, :], batch[:, 1:, :]
        output, _ = self(X)
        assert isinstance(output, Tensor)
        Y = self.fc(output.t())
        return self.loss(Y, y)
    
    def training_epoch_end(self, outputs: dict) -> None:
        train_loss = torch.stack([o['loss'] for o in outputs]).mean()
        train_ppl = torch.exp(train_loss)
        self.log_dict({'train_loss': train_loss, 'train_ppl': train_ppl})
        
    def validation_step(self, batch, _) -> dict:
        X, y = batch[:, :-1, :], batch[:, 1:, :]
        output, _ = self(X)
        assert isinstance(output, Tensor)
        Y = self.fc(output.t())
        return {'loss': self.loss(Y, y)}
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([o['loss'] for o in outputs]).mean()
        val_ppl = torch.exp(val_loss)
        self.log_dict({'val_loss': val_loss, 'val_ppl': val_ppl})
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self.lr)
    