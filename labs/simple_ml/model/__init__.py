import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils import data
import datasets


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

class Vocab:
    def __init__(self, bow: set):
        sorted_bow = sorted(bow)
        self.enc_dict = dict()
        self.dec_dict = dict()
        for index, c in enumerate(sorted_bow):
            self.enc_dict[c] = index
            self.dec_dict[index] = c
    
    def __getitem__(self,index):
        return self.enc_dict[index]
    
    def decode(self, index):
        return self.dec_dict[index]
    
    
class NullTrimmer():
    def __init__(self, max_length):
        self.max_length = max_length
        
    def __call__(self, s: str):
        pad_count = self.max_length - len(s)
        if pad_count > 0:
            return s + ''.join([' ' for _ in range(pad_count)])
        elif pad_count < 0:
            return s[:self.max_length]
        else:
            return s
    
class WikiDataset(data.Dataset):
        
    def __init__(self, n_steps:int, target:str='train'):
        super().__init__()
        self.n_steps = n_steps
        assert target in {"train", "test", "validate"}
        data = datasets.load_dataset('wikitext', 'wikitext-2-v1')[target]
        chars = set()
        for line in data['text']:
            chars = chars.union(set(line))
            
        self.vocab = Vocab(chars)
        trimmer = NullTrimmer(n_steps)
        data = data.map(lambda x: {'trimmed':trimmer(x)}, input_columns='text')
        self.data = data.map(lambda x: {'emb':[self.vocab[c] for c in x]}, input_columns='trimmed')
        self.data.set_format('torch', columns=['emb', 'text'])
        print(self.data[0])
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
        
        
        
        
        
