import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from model import GRUModule, EncoderDecoder
from data import Korean, English, PaddingCollator, get_default_tokenizer, KoEnParallel, ParallelCollator
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers
from torch.utils import data

LANG_CHOICES = [
    'ko', 'en'
]

def train_generative_task(args):
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
    
    trainer = pl.Trainer(logger=logger, 
                         callbacks=[ckpt_callback], 
                         accelerator='gpu', 
                         max_epochs=args.max_epochs)
    
    trainer.fit(encoder, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
def train_seq2seq(args):
    ko_tokenizer = get_default_tokenizer('ko')
    en_tokenizer = get_default_tokenizer('en')
    
    train_dataset = KoEnParallel('train', include_para_pat=False)
    val_dataset = KoEnParallel('validation')
    seq2seq = EncoderDecoder(lr=args.lr,freeze=False)
    
    collator = ParallelCollator(from_tokenizer=ko_tokenizer,
                     to_tokenizer=en_tokenizer, 
                     from_pad_id=ko_tokenizer.encode('<pad>')[0], 
                     to_pad_id=en_tokenizer.encode('<pad>')[0], 
                     from_eos_id=ko_tokenizer.encode('<eos>')[0], 
                     to_bos_id=en_tokenizer.encode('<bos>')[0], 
                     to_eos_id=en_tokenizer.encode('<eos>')[0])
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=collator, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=collator)
    
    ckpt_callback = callbacks.ModelCheckpoint(f'model/{args.task}',
                                              filename='model-{epoch}-{val_loss:.2f}', 
                                              mode='min', 
                                              monitor='val_loss', 
                                              save_top_k=3)
    logger = loggers.TensorBoardLogger('logs')
    
    trainer = pl.Trainer(logger=logger, 
                         callbacks=[ckpt_callback],
                         accelerator='gpu',
                         max_epochs=args.max_epochs)
    
    trainer.fit(seq2seq, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    

def main(args):
    if args.task == 'gen':
        train_generative_task(args)
    elif args.task == 'seq2seq':
        train_seq2seq(args)
    
    
    
    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('train.py')
    arg_parser.add_argument('--task', type=str, choices=['gen','seq2seq'], default='gen')
    arg_parser.add_argument('--lang', type=str, choices=['ko','en'], default='ko')
    arg_parser.add_argument('--max_epochs', type=int, default=10)
    arg_parser.add_argument('--batch', type=int, default=10)
    arg_parser.add_argument('--max_steps', type=int, default=1024)
    arg_parser.add_argument('--lr', type=float, default=1e-4)
    main(arg_parser.parse_args())