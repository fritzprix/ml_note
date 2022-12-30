
import sys
from torchvision import datasets
from torch import utils
from torch.utils import data
from models import SimpleFashionMNISTClassifier, Preprocess, ResNetBasedClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import callbacks
import wandb
import argparse

def main(args):
    
    wandb.init()
    dataset = datasets.FashionMNIST('./data', train=True, transform=Preprocess, download=True)
    train_dataset, val_dataset = data.random_split(dataset, [0.9, 0.1])
    print(f"data split (train : validation) = ({len(train_dataset)}:{len(val_dataset)})")
    
    train_dataloader = data.DataLoader(train_dataset, args.batch_size)
    val_dataloader = data.DataLoader(val_dataset, args.batch_size)
    if args.model == "simple":
        model = SimpleFashionMNISTClassifier(len(dataset.classes), lr=args.lr)
    elif args.model == "resnet":
        model = ResNetBasedClassifier(28, len(dataset.classes), lr=args.lr)
    logger = WandbLogger(log_model='all')
    ckpt_callback = callbacks.ModelCheckpoint(monitor='val_acc',mode='max')
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    
    
    trainer = pl.Trainer(logger=logger, callbacks=[ckpt_callback, early_stop_callback], accelerator='gpu', max_epochs=args.max_epochs)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    
if __name__ == '__main__':
    argparse = argparse.ArgumentParser("train.py")
    argparse.add_argument("--is_base_frozen", type=bool, default=False)
    argparse.add_argument("--model", type=str, default="simple")
    argparse.add_argument("--max_epochs", type=int, default=100)
    argparse.add_argument("--batch_size", type=int, default=128)
    argparse.add_argument("--lr", type=float, default=5e-05)
    main(argparse.parse_args())