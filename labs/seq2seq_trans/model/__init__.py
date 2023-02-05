import pytorch_lightning as pl
import torch
import transformers
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
                          num_layers=num_layers)
        
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
    
    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, gen_len: int, tokenize: transformers.AutoTokenizer, warmup=True):
        # prefix comes in with shape of (batch, seq) with token_id
        print(f"prefix : {prefix}")
        prefix_sparsev = self.embedding(prefix.t().type(int64))
        # prefix_sparsev encoded from transpose of prefix, the shape => (seq, batch, emb)
        X, state = prefix_sparsev[0,:,:].unsqueeze(0) if warmup else prefix_sparsev, None
        prefix_seq_len = prefix.shape[1]
        for i in range(1, gen_len):
            output, state = self.enc(X, state)
            assert isinstance(output, torch.Tensor)
            # output (seq, batch, 2 * hidden_size)
            # output (seq, batch, input_size)
            if i < prefix_seq_len:
                y = prefix_sparsev[i, :, :].unsqueeze(0)
            else:
                y = self.embedding(self.fc(output).argmax(dim=2))
            assert isinstance(y, torch.Tensor)
            X = torch.cat([X, y[-1,:,:].unsqueeze(0)], dim=0)
        rev_emb = X @ self.embedding.weight.T
        return tokenize.decode(rev_emb.argmax(dim=2).reshape(-1).detach().tolist())
        
    def training_step(self, batch, _) -> Tensor:
        X, y = batch[:, :-1], batch[:, 1:]
        output, _ = self(X)
        assert isinstance(output, Tensor)
        Y = self.fc(output.transpose(0, 1))
        return self.loss(Y.reshape(-1, Y.shape[-1]), y.reshape(-1).long())
    
    def training_epoch_end(self, outputs: dict) -> None:
        train_loss = torch.stack([o['loss'] for o in outputs]).mean()
        train_ppl = torch.exp(train_loss)
        self.log_dict({'train_loss': train_loss, 'train_ppl': train_ppl})
        
    def validation_step(self, batch, _) -> dict:
        X, y = batch[:, :-1], batch[:, 1:]
        output, _ = self(X)
        assert isinstance(output, Tensor)
        Y = self.fc(output.transpose(0, 1))
        return {'loss': self.loss(Y.reshape(-1, Y.shape[-1]), y.reshape(-1).long())}
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([o['loss'] for o in outputs]).mean()
        val_ppl = torch.exp(val_loss)
        self.log_dict({'val_loss': val_loss, 'val_ppl': val_ppl})
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), self.lr)
    
def get_default_tokenizer(lang:str):
    lang = lang.lower()
    assert lang in ["ko", "en"]
    if lang == "ko":
        return transformers.AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
    else:
        return transformers.AutoTokenizer.from_pretrained("gpt2")

class EnDecoder(pl.LightningModule):
    def __init__(self, input_size:int, embed_size, hidden_size:int, num_layers:int=2, lr:float=1e-4, padding_id=0) -> None:
        super().__init__()
        self.save_hyperparameters()
    