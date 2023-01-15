from model import FFNAutoregressor, GRUAutoregress, SinusoidalTimeSeries
import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch.utils import data
import argparse
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

        

def main(args: argparse.Namespace):
    dataset = SinusoidalTimeSeries(0, n_samples=4000, n_steps=args.steps)
    train(args.model, args.steps, args.lr, dataset, args.batch,args.max_epoch, args.device)

    
def train(model_name:str, n_steps, lr, dataset, batch_size, max_epoch, device: str = 'cpu'):
   
    if model_name == 'ffn':
        model = FFNAutoregressor(lr, n_steps, 1, 2,num_hidden=3)
    elif model_name == 'gru':
        model = GRUAutoregress(1, num_layers=3, lr=lr)
        
    if device == 'cpu':
        num_workers = 1
    else:
        num_workers = os.cpu_count()
    
             
    # train_dataset, test_dataset, val_dataset = dataset, dataset[3000:3700], dataset[3000:4000]
    train_dataset, test_dataset, val_dataset = data.random_split(dataset, [0.5, 0.3, 0.2], torch.Generator().manual_seed(6))
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, check_finite=True)
    model_ckpt = callbacks.ModelCheckpoint(f"./model/{model_name}", 
                                        filename='model-{epoch}-{val_loss:.2f}', 
                                        monitor='val_loss', mode='min',
                                        save_top_k=3)
    
    trainer = pl.Trainer(callbacks=[model_ckpt, early_stop], max_epochs=max_epoch, accelerator=device)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('train.py')
    args_parser.add_argument('--model',choices=['ffn', 'gru'], type=str, default='ffn')
    args_parser.add_argument('--lr', type=float, default=1e-4)
    args_parser.add_argument('--steps', type=int, default=10)
    args_parser.add_argument('--batch', type=int, default=10)
    args_parser.add_argument('--max_epoch', type=int, default=100)
    args_parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    args_parser.add_argument('--split_summary', type=bool, default=False)
    args: argparse.Namespace = args_parser.parse_args()
    main(args)