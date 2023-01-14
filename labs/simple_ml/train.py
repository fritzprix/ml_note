from model import GRUML
from argparse import ArgumentParser, Namespace
import datasets
from transformers import BookCorpus


def main(args: Namespace):
    GRUML(vocab_size=10)
    book_dataset: datasets.DatasetDict = datasets.load_dataset("bookcorpus")
    train_dataset = book_dataset['train']
    print(next(iter(train_dataset))['text'])
    


if __name__ == '__main__':
    arg_parser = ArgumentParser('train.py')
    main(arg_parser.parse_args())