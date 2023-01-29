from model import GRUML, WikiDataset, TimeMachine, RNNML
from argparse import ArgumentParser, Namespace
from torch.utils import data
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning import callbacks
import os
from torchtext.vocab import Vocab
from d2l import torch as d2l



def main(args: Namespace):
    d2l.read_time_machine()
    if args.log == 'tf':
        logger = loggers.TensorBoardLogger('tf_logs')
    else:
        logger = loggers.WandbLogger(project='simple_glm_character')
    if args.data == 'timemachine':
        tm_dataset = TimeMachine(n_steps=args.n_steps)
        train_dataset, val_dataset = data.random_split(tm_dataset, [0.9, 0.1])
        vocab = tm_dataset.vocab
    else:
        train_dataset = WikiDataset(n_steps=args.n_steps, target='train')
        val_dataset = WikiDataset(n_steps=args.n_steps, target='validation')
        vocab = train_dataset.vocab
    assert isinstance(vocab, Vocab)
    
    print([vocab.lookup_token(i) for i in range(len(vocab))])
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch, 
                                       num_workers=os.cpu_count(), shuffle=True)
    
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=args.batch, 
                                     num_workers=os.cpu_count())
    
    input_size = len(vocab)
    padding_id = vocab['<pad>']
    model = GRUML(input_size=input_size, lr=args.lr, num_hidden=input_size * 8, padding_id=padding_id)
    
    ckpt_callback = callbacks.ModelCheckpoint(f'./model/gru_ml/{args.data}',
                                              filename='model-{epoch}-{val_loss:.3f}', 
                                              save_top_k=3, 
                                              monitor='val_loss', 
                                              mode='min')
    
    early_callback = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    
    trainer = pl.Trainer(logger=logger, 
                         max_epochs=args.max_epoch, 
                         callbacks=[ckpt_callback, early_callback], 
                         gradient_clip_val=1,
                         accelerator='gpu', 
                         check_val_every_n_epoch=2)
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    torch.save(vocab, f'./model/gru_ml/{args.data}/vocab.pt')
    
    
    

if __name__ == '__main__':
    arg_parser = ArgumentParser('train.py')
    arg_parser.add_argument('--data', choices=['timemachine', 'wiki'], default='wiki'),
    arg_parser.add_argument('--n_steps', type=int, default=10)
    arg_parser.add_argument('--batch', type=int, default=40)
    arg_parser.add_argument('--lr', type=float, default=1e-4)
    arg_parser.add_argument('--max_epoch', type=int, default=100)
    arg_parser.add_argument('--log', choices=['tf', 'wandb'], default='tf')
    main(arg_parser.parse_args())