import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchtext import vocab, functional as TF
from torchtext import transforms
import torchtext
import datasets


class GRUML(pl.LightningModule):
    def __init__(self, input_size, lr:float = 1e-4,  num_hidden=40, num_layers=3, use_sliding=True, padding_id=0):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.padding_id = padding_id
        self.lr = lr
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=num_hidden, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(num_hidden, input_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.padding_id)
        self.use_sliding = use_sliding
        
        
    def forward(self,X):
        emb = F.one_hot(X, self.input_size).float()
        out, _ = self.gru(emb)
        return self.fc(out)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        seq = batch['seqs']
        losses = []
        for subsq_len in range(2, seq.shape[1] - 1, 1):
            X = seq[:, 0: subsq_len]
            y = seq[:, 1: subsq_len + 1]
            y_hat = self(X)
            assert isinstance(y_hat, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            loss = self.loss(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
            losses.append(loss)
        return torch.stack(losses).mean()
    
    def training_epoch_end(self, outputs):
        avg = torch.stack([o['loss'] for o in outputs]).mean()
        self.log("train_loss", avg)
        
    
    def validation_step(self, batch, _):
        seq = batch['seqs']
        X, y = seq[:, 0: -2], seq[:, 1:-1]
        y_hat = self(X)
        assert isinstance(y_hat, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        loss = self.loss(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
        return {"loss": loss}
    
    def validation_epoch_end(self, outputs: list[dict]):
        avg_loss = torch.stack([o['loss'] for o in outputs]).mean()
        self.log("val_loss", avg_loss)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self.lr)
        

    
class WikiDataset(data.Dataset):
        
    def __init__(self, n_steps:int, target:str='train'):
        super().__init__()
        self.n_steps = n_steps
        assert target in {"train", "test", "validation"}
        data = datasets.load_dataset('wikitext', 'wikitext-2-v1')[target]
        chars = set()
        for line in data['text']:
            chars = chars.union(set(line))
        
        self.vocab = vocab.build_vocab_from_iterator(chars, specials=['<pad>', '<unk>'])
        
        self.data = data.filter(lambda x: len(x) > 0, input_columns='text') \
                .map(lambda x: {'text': x.strip()}, input_columns='text') \
                .map(lambda x: {"token_id": [self.vocab[c] for c in x]}, input_columns='text') \
                .map(lambda x: {"token_id": TF.truncate(x, n_steps)}, input_columns='token_id') \
                .map(lambda x: {"length": len(x)}, input_columns='token_id')
                
        self.data.set_format('torch', columns=['text', 'length', 'token_id'])
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collater(self):
        return BatchPaddingCollater(self.vocab['<pad>'])
        

class BatchPaddingCollater:
    def __init__(self, padding_val) -> None:
        self.padding_val = padding_val
        
    def __call__(self, batch):
        seqs = TF.pad_sequence([e['token_id'] for e in batch], batch_first=True, padding_value= self.padding_val)
        lengs = torch.Tensor([e['length'] for e in batch])
        return {'seqs': seqs, 'lens':lengs}
        
        
        
