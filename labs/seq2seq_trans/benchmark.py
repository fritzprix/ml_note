import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from torch.utils import data
from model import EncoderDecoder
from data import KoEnParallel, get_default_tokenizer, ParallelCollator
import pytorch_lightning as pl
from pytorch_lightning import loggers


def bm_tranlation(args):
    logger = loggers.TensorBoardLogger('logs')
    val_dataset = KoEnParallel('validation')
    seq2seq = EncoderDecoder.load_from_checkpoint('model/seq2seq/model.ckpt')
    seq2seq.freeze()
    ko_tokenizer = get_default_tokenizer('ko')
    en_tokenizer = get_default_tokenizer('en')
    
    collator = ParallelCollator(from_tokenizer=ko_tokenizer,
                     to_tokenizer=en_tokenizer, 
                     from_pad_id=ko_tokenizer.encode('<pad>')[0], 
                     to_pad_id=en_tokenizer.encode('<pad>')[0], 
                     from_eos_id=ko_tokenizer.encode('<eos>')[0], 
                     to_bos_id=en_tokenizer.encode('<bos>')[0], 
                     to_eos_id=en_tokenizer.encode('<eos>')[0])
    
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=collator)
    
    trainer = pl.Trainer(logger=logger, accelerator='gpu')
    trainer.validate(seq2seq, dataloaders=val_dataloader)
    
    

def bm_gen(args):
    pass

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('benchmark.py')
    arg_parser.add_argument('--task', type=str, choices=['trans', 'gen'], default='trans')
    arg_parser.add_argument('--lang', type=str, default='ko')
    arg_parser.add_argument('--batch', type=int, default=10)
    args = arg_parser.parse_args()
    if args.task == 'trans':
        bm_tranlation(args)
    elif args.task == 'gen':
        bm_gen(args)