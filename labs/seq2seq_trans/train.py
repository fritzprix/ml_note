import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from model import GRUEncoder, get_default_tokenizer
from data import Korean, English
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers
from torch.utils import data

LANG_CHOICES = [
    'ko', 'en'
]

def main(args):
    assert args.lang.lower() in LANG_CHOICES
    tokenizer = get_default_tokenizer(args.lang)
    if args.lang == 'ko':
        train_dataset = Korean(tokenizer, 1000, 'train')
        val_dataset = Korean(tokenizer, 1000, 'validation')
    else:
        train_dataset = English(tokenizer, 1000, 'train')
        val_dataset = English(tokenizer, 1000, 'validation')
    
    encoder = GRUEncoder(tokenizer.vocab_size, 128, 128, padding_id=tokenizer.pad_token_id)
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=os.cpu_count(), collate_fn=train_dataset)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=val_dataset)
    
    logger = loggers.TensorBoardLogger('logs')
    ckpt_callback = callbacks.ModelCheckpoint(f'model/encoder/{args.lang}',
                                              filename='model-{epoch}-{val_loss:.2f}', 
                                              mode='min', 
                                              monitor='val_loss', 
                                              save_top_k=3)
    
    trainer = pl.Trainer(logger=logger, callbacks=[ckpt_callback], accelerator='gpu', max_epochs=args.max_epochs)
    trainer.fit(encoder, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    
    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('train.py')
    arg_parser.add_argument('--lang', type=str, choices=['ko','en'], default='ko')
    arg_parser.add_argument('--max_epochs', type=int, default=10)
    arg_parser.add_argument('--batch', type=int, default=10)
    main(arg_parser.parse_args())