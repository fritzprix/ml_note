import pytorch_lightning as pl
import torch
import transformers
from torch import nn, Tensor, int64



class GRUModule(pl.LightningModule):
    def __init__(self, input_size:int, embed_size:int, hidden_size:int, num_layers:int=2, lr:float=1e-4, padding_id=0) -> None:
        super(GRUModule, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.padding_id = padding_id
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=padding_id)
        self.gru = nn.GRU(input_size=embed_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers)
        
        self.fc = nn.LazyLinear(input_size) # linear layer for CLM training
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_id)
    
    def forward(self, X, state=None):
        # X (N,L,IV) => Torch RNN (L,N,IV)
        assert isinstance(X, Tensor)
        emb_X = self.embedding(X.t().type(int64))
        outputs, state = self.gru(emb_X, state)
        # output will (L,N,2 * hidden_size)
        # state will (2 * num_layers, hidden_size)
        return outputs, state
    
    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, gen_len: int, tokenizer: transformers.AutoTokenizer, warmup=True):
        # prefix comes in with shape of (batch, seq) with token_id
        prefix_sparsev = self.embedding(prefix.t().type(int64))
        # prefix_sparsev encoded from transpose of prefix, the shape => (seq, batch, emb)
        X, state = prefix_sparsev[0,:,:].unsqueeze(0) if warmup else prefix_sparsev, None
        prefix_seq_len = prefix.shape[1]
        for i in range(1, gen_len):
            output, state = self.gru(X, state)
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
        return tokenizer.decode(rev_emb.argmax(dim=2).reshape(-1).detach().tolist())
        
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
    
    


class Encoder(nn.Module):
    def __init__(self, model: GRUModule):
        super().__init__()
        self.model = model
    
    def forward(self, X, *args):
        outputs, state = self.model(X, *args)
        return outputs, state
    
class Decoder(nn.Module):
    def __init__(self, model: GRUModule) -> None:
        super().__init__()
        self.model = model
        
    
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs
    
    def forward(self, X, state=None):
        embs = self.model.embedding(X.t().long())
        # enc_output will (L,N,2 * hidden_size)
        # hidden_state will (2 * num_layers, hidden_size)
        enc_output, hidden_state = state
        context = enc_output[-1] # value of last step will be used as context
        assert isinstance(context, torch.Tensor)
        context = context.repeat([embs.shape[0], 1, 1])
        embs_and_context = torch.cat((embs, context), -1)
        outputs, hidden_state = self.model.gru(embs_and_context, hidden_state)
        
        return outputs, [enc_output, hidden_state]


class EncoderDecoder(pl.LightningModule):
    def __init__(self, encoder: Encoder=None, decoder:Decoder=None,lr:float=1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        if encoder is None:
            pre_trained = GRUModule.load_from_checkpoint('model/ko/model.ckpt')
            pre_trained.freeze()
            encoder = Encoder(pre_trained)
        
        if decoder is None:
            pre_trained = GRUModule.load_from_checkpoint('model/en/model.ckpt')
            pre_trained.freeze()
            decoder = Decoder(pre_trained)
            
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        
    def forward(self, enc_X, dec_X, *args):
        enc_all_output = self.encoder(enc_X)
        init_state = self.decoder.init_state(enc_all_outputs=enc_all_output)
        output, state = self.decoder(dec_X, init_state)
        return output, state
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
