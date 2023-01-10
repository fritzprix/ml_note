from model import FFNAutoregressor, GRUAutoregress, SinusoidalTimeSeries
import argparse


def main(args: argparse.Namespace):
    if args.model == 'ffn':
        model = FFNAutoregressor(args.lr, args.steps, 1, 4)
    elif args.model == 'gru':
        model = GRUAutoregress(1)
    dataset = SinusoidalTimeSeries()
    for batch in iter(dataset):
        print(batch)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('train.py')
    args_parser.add_argument('--model',choices=['ffn', 'gru'], type=str, default='ffn')
    args_parser.add_argument('--lr', type=float, default=1e-4)
    args_parser.add_argument('--steps', type=int, default=10)
    args: argparse.Namespace = args_parser.parse_args()
    main(args)