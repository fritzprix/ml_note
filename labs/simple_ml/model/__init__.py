import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchtext import vocab
import datasets
from d2l import torch as d2l
import re

MIN_NGRAM = 2

class RNNML(pl.LightningModule):
    def __init__(self, input_size, lr:float = 1e-4,  num_hidden=40, num_layers=10, use_sliding=False, padding_id=0):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.padding_id = padding_id
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=num_hidden, num_layers=num_layers)
        self.fc = nn.Linear(num_hidden, input_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_id)
        self.lr = lr
        self.use_sliding = use_sliding
    
    def forward(self, X, hidden=None) -> torch.Tensor:
        # X will be (N, L, 1) or (L, 1)
        emb = F.one_hot(X, self.input_size).float()
        out, hidden = self.rnn(emb, hidden)
        return self.fc(out), hidden

    def training_step(self, batch, _) -> torch.Tensor:
        seq = batch
        losses = []
        sliding = range(MIN_NGRAM, seq.shape[1] - 1, 2) if self.use_sliding else range(seq.shape[1] - 1, seq.shape[1])
        X = seq[:, 0: -1]
        y = seq[:, 1:]
        y_hat, _ = self(X)
        assert isinstance(y_hat, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        return self.loss(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
    
    def training_epoch_end(self, outputs):
        avg = torch.stack([o['loss'] for o in outputs]).mean()
        ppl = torch.exp(avg)
        self.log("train_ppl", ppl)
        self.log("train_loss", avg)
        
    def validation_step(self, batch, _):
        seq = batch
        X, y = seq[:, 0: -1], seq[:, 1:]
        y_hat, _ = self(X)
        assert isinstance(y_hat, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        loss = self.loss(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
        return {"loss": loss}
    
    def validation_epoch_end(self, outputs: list[dict]):
        avg_loss = torch.stack([o['loss'] for o in outputs]).mean()
        ppl = torch.exp(avg_loss)
        self.log("val_ppl", ppl)
        self.log("val_loss", avg_loss)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self.lr)

class GRUML(pl.LightningModule):
    def __init__(self, input_size, lr:float = 1e-4,  num_hidden=40, num_layers=10, use_sliding=False, padding_id=0):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.padding_id = padding_id
        
        self.lr = lr
        self.gru = nn.GRU(input_size=input_size, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(num_hidden, input_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=padding_id)
        self.use_sliding = use_sliding
        
    def forward(self, X):
        emb = F.one_hot(X, self.input_size).float()
        out, _ = self.gru(emb)
        return self.fc(out)
    
    def training_step(self, batch, _) -> torch.Tensor:
        seq = batch
        losses = []
        sliding = range(MIN_NGRAM, seq.shape[1] - 1, 2) if self.use_sliding else range(seq.shape[1] - 1, seq.shape[1])
        for subsq_len in sliding:
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
        ppl = torch.exp(avg)
        self.log("train_ppl", ppl)
        self.log("train_loss", avg)
        
    def validation_step(self, batch, _):
        seq = batch
        X, y = seq[:, 0: -1], seq[:, 1:]
        y_hat = self(X)
        assert isinstance(y_hat, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        loss = self.loss(y_hat.reshape(-1, y_hat.shape[-1]), y.reshape(-1))
        return {"loss": loss}
    
    def validation_epoch_end(self, outputs: list[dict]):
        avg_loss = torch.stack([o['loss'] for o in outputs]).mean()
        ppl = torch.exp(avg_loss)
        self.log("val_ppl", ppl)
        self.log("val_loss", avg_loss)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self.lr)
        
class TimeMachine(data.Dataset):
    def __init__(self, n_steps:int, max_tokens:int=10000):
        super().__init__()
        self.n_steps = n_steps
        data, voc = d2l.load_data_time_machine(1, num_steps=n_steps)
        corpus = data.corpus
        self.vocab = vocab.vocab(voc.token_to_idx, specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.data = torch.Tensor([corpus[i:i + n_steps] for i in range(len(corpus) - n_steps)]).long()
        
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)
    

    
class WikiDataset(data.Dataset):
        
    def __init__(self, n_steps:int, target:str='train'):
        super().__init__()
        self.n_steps = n_steps
        assert target in {"train", "test", "validation"}
        data = datasets.load_dataset('wikitext', 'wikitext-2-v1')[target]
        chars = set()
        lines = []
        for line in data['text']:
            line = re.sub('[^a-zA-Z1-9.,]+',' ',line).lower()
            line = re.sub('[\s]+', ' ', line).strip()
            lines.append(line)
            chars = chars.union(set(line))
        
        
        self.vocab = vocab.build_vocab_from_iterator(chars, specials=['<pad>', '<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        data = ' '.join(lines)
        data = [list(data[i:i + n_steps]) for i in range(len(data) - n_steps)]
        self.data = torch.Tensor([self.vocab.lookup_indices(seq) for seq in data]).long()
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
        
        
