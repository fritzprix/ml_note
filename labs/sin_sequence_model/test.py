from model import FFNAutoregressor, GRUAutoregress, SinusoidalTimeSeries
from argparse import Namespace, ArgumentParser


def main(args: Namespace):
    
    ffn_model = FFNAutoregressor.load_from_checkpoint('./model/fnn/model.ckpt')
    gru_model = GRUAutoregress.load_from_checkpoint('./model/gru/model.ckpt')
    


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    main(arg_parser.parse_args())
    
