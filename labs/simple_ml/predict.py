import torch
from model import GRUML, WikiDataset, TimeMachine
from argparse import Namespace, ArgumentParser
from inspect import getmembers

def main(args):
    assert isinstance(args.prompt, str)
    vocab = torch.load(f'./model/gru_ml/{args.data}/vocab.pt')
    model = GRUML.load_from_checkpoint(f'./model/gru_ml/{args.data}/model.ckpt')
    model.cuda()
    model.freeze()
    pred = model.predict(args.prompt.lower(), args.len, vocab)
    print(''.join(pred))
        
    

if __name__ == '__main__':
    argparser = ArgumentParser('predict.py')
    argparser.add_argument('--prompt', type=str, default='hello')
    argparser.add_argument('--len', type=int, default=20)
    argparser.add_argument('--data', choices=['timemachine', 'wiki'], default='wiki')
    main(argparser.parse_args())