from model import GRUML, WikiDataset
from argparse import ArgumentParser, Namespace
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning import callbacks
import wandb
import os




def main(args: Namespace):
    
    wandb_logger = loggers.WandbLogger(project='simple_glm_character')
    train_dataset = WikiDataset(n_steps=40, target='train')
    val_dataset = WikiDataset(n_steps=40, target='validation')
    collate_fn = train_dataset.collater()
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=collate_fn)
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=collate_fn)
    
    model = GRUML(input_size=len(train_dataset.vocab), lr=args.lr, padding_id=train_dataset.vocab['<pad>'])
    
    ckpt_callback = callbacks.ModelCheckpoint('./model/gru_ml/', 
                                              filename='model-{epoch}-{val_loss:.3f}', 
                                              save_top_k=3, 
                                              monitor='val_loss', 
                                              mode='min')
    
    early_callback = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=args.max_epoch, callbacks=[ckpt_callback, early_callback], accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    
    
    
    


if __name__ == '__main__':
    arg_parser = ArgumentParser('train.py')
    arg_parser.add_argument('--batch', type=int, default=40)
    arg_parser.add_argument('--lr', type=float, default=1e-4)
    arg_parser.add_argument('--max_epoch', type=int, default=100)
    main(arg_parser.parse_args())