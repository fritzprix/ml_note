import torch
from model import GRUML, WikiDataset, TimeMachine
from argparse import Namespace, ArgumentParser
from inspect import getmembers

def main(args):
    if args.data == 'timemachine':
        dataset = TimeMachine(n_steps=10)
    else:
        dataset = WikiDataset(n_steps=10)
    vocab = dataset.vocab
    model = GRUML.load_from_checkpoint(f'./model/gru_ml/{args.data}/model.ckpt')
    seq = [dataset.vocab[c] for c in args.prompt]
    X = torch.Tensor(seq).long()
    for _ in range(args.len):
        X = model(X).argmax(dim=1)
        seq.append(X[-1].item())
    
    answer = ''.join([vocab.lookup_token(id) for id in seq])
    print(answer)
        
    
    
    

if __name__ == '__main__':
    argparser = ArgumentParser('predict.py')
    argparser.add_argument('--prompt', type=str, default='hello')
    argparser.add_argument('--len', type=int, default=20)
    argparser.add_argument('--data', choices=['timemachine', 'wiki'], default='wiki')
    main(argparser.parse_args())