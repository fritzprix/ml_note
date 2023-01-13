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


def extract_labels(dataset: data.Dataset) -> torch.Tensor:
    return torch.cat([y for _,y in iter(dataset)])

def plot_datasplit(whole: data.Dataset, subsets: list[data.Dataset], colors: list[str], labels: list[str]):
    assert isinstance(whole, data.Dataset)
    _ = plt.figure(figsize=(10,10))
    y = extract_labels(dataset=whole).numpy()
    subsets = [extract_labels(subset).numpy() for subset in subsets]
    for i, lv in tqdm(enumerate(y)):
        color = 'k'
        for sub, c in zip(subsets, colors):
            if lv in sub:
                color = c
        plt.scatter(x=i, y=lv, c=color, s=5)
    legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(subsets))]
    plt.legend(handles=legend_elements)
    plt.show()
        

def main(args: argparse.Namespace):
    if args.model == 'ffn':
        model = FFNAutoregressor(args.lr, args.steps, 1, 2,num_hidden=3)
    elif args.model == 'gru':
        model = GRUAutoregress(1, num_layers=3)
    
    
    if args.device == 'cpu':
        num_workers = 1
    else:
        num_workers = os.cpu_count()
    
    dataset = SinusoidalTimeSeries(n_samples=4000)
    train_dataset, test_dataset, val_dataset = data.random_split(dataset, [0.4, 0.4, 0.2], generator=torch.Generator().manual_seed(6))
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch, num_workers=num_workers)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch, num_workers=num_workers)
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, check_finite=True)
    model_ckpt = callbacks.ModelCheckpoint(f"./model/{args.model}", 
                                        filename='model-{epoch}-{val_loss:.2f}', 
                                        monitor='val_loss', mode='min',
                                        save_top_k=3)
    
    trainer = pl.Trainer(callbacks=[early_stop, model_ckpt], max_epochs=args.max_epoch, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    
    print("now plotting split summary...")
    if args.split_summary:
        plot_datasplit(dataset, 
                       subsets=[train_dataset, val_dataset, test_dataset], 
                       colors=['r', 'g', 'b'], 
                       labels=['train', 'validation', 'test'])
    
def train(model_name, model, dataset, batch_size, num_workers, max_epoch):
    train_dataset, test_dataset, val_dataset = data.random_split(dataset, [0.4, 0.4, 0.2], generator=torch.Generator().manual_seed(6))
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, check_finite=True)
    model_ckpt = callbacks.ModelCheckpoint(f"./model/{model_name}", 
                                        filename='model-{epoch}-{val_loss:.2f}', 
                                        monitor='val_loss', mode='min',
                                        save_top_k=3)
    
    trainer = pl.Trainer(callbacks=[early_stop, model_ckpt], max_epochs=max_epoch, accelerator='gpu')
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