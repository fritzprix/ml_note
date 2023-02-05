import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from model import KorEncoder
from data import Korean
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers
from torch.utils import data


def main(args):
    ko_tokenizer = AutoTokenizer.from_pretrained('skt/ko-gpt-trinity-1.2B-v0.5')
    ko_train_dataset = Korean(ko_tokenizer, 1000, 'train')
    ko_val_dataset = Korean(ko_tokenizer, 1000, 'validation')
    
    ko_encoder = KorEncoder(ko_tokenizer.vocab_size, 128, 128, padding_id=ko_tokenizer.pad_token_id)
    
    ko_train_dataloader = data.DataLoader(ko_train_dataset, batch_size=args.batch, shuffle=True, num_workers=os.cpu_count(), collate_fn=ko_train_dataset)
    ko_val_dataloader = data.DataLoader(ko_val_dataset, batch_size=args.batch, num_workers=os.cpu_count(), collate_fn=ko_val_dataset)
    
    logger = loggers.TensorBoardLogger('logs')
    ckpt_callback = callbacks.ModelCheckpoint('model/encoder/ko', 
                                              filename='model-{epoch}-{val_loss:.2f}', 
                                              mode='min', 
                                              monitor='val_loss', 
                                              save_top_k=3)
    
    trainer = pl.Trainer(logger=logger, callbacks=[ckpt_callback], accelerator='gpu', max_epochs=args.max_epochs)
    trainer.fit(ko_encoder, train_dataloaders=ko_train_dataloader, val_dataloaders=ko_val_dataloader)
    
    
    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('train.py')
    arg_parser.add_argument('--max_epochs', type=int, default=10)
    arg_parser.add_argument('--batch', type=int, default=10)
    main(arg_parser.parse_args())