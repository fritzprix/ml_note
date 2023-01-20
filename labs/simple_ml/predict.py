import torch
from model import GRUML, WikiDataset
from argparse import Namespace, ArgumentParser


def main(args):
    dataset = WikiDataset(n_steps=40)
    vocab = dataset.vocab
    model = GRUML.load_from_checkpoint('./model/gru_ml/model.ckpt')
    seq = [dataset.vocab[c] for c in args.prompt]
    X = torch.Tensor(seq).long()
    for _ in range(args.len):
        print(X)
        X = model(X).argmax(dim=1)
        seq.append(X[-1].item())
    answer = str([vocab.lookup_token(id) for id in seq])
    print(answer)
        
    
    
    

if __name__ == '__main__':
    argparser = ArgumentParser('predict.py')
    argparser.add_argument('--prompt', type=str, default='hello')
    argparser.add_argument('--len', type=int, default=20)
    main(argparser.parse_args())