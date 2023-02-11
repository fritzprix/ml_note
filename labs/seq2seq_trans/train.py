import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from model import GRUModule
from data import Korean, English, PaddingCollator, get_default_tokenizer
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
    pad_id = tokenizer.encode('<pad>')[0]
    collator = PaddingCollator(args.max_steps, tokenizer=tokenizer, pad_id=pad_id)
    if args.lang == 'ko':
        train_dataset = Korean('train')
        val_dataset = Korean('validation')
    else:
        train_dataset = English('train')
        val_dataset = English('validation')
    
    encoder = GRUModule(tokenizer.vocab_size, 128, 128, padding_id=pad_id)
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=os.cpu_count(), collate_fn=collator)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=collator)
    
    logger = loggers.TensorBoardLogger('logs')
    ckpt_callback = callbacks.ModelCheckpoint(f'model/{args.lang}',
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
    arg_parser.add_argument('--max_steps', type=int, default=1024)
    main(arg_parser.parse_args())