import argparse
import model
import data
import torch

LANG_CHOICE = set(["ko","en"])

def generate(args):
    assert args.lang in LANG_CHOICE
    tokenizer = data.get_default_tokenizer(args.lang)
    encoder = model.GRUModule.load_from_checkpoint(f'model/{args.lang}/model.ckpt')
    cuda = torch.device('cuda')
    tokens = tokenizer.encode(args.prompt, return_tensors="pt").to(cuda)
    encoder = encoder.cuda()
    print(encoder.generate(tokens, args.len, tokenizer))
    
def translate(args):
    ko_tokenizer = data.get_default_tokenizer('ko')
    en_tokenizer = data.get_default_tokenizer('en')
    
    seq2seq = model.EncoderDecoder.load_from_checkpoint('model/seq2seq/model.ckpt')
    seq2seq.freeze()
    ko_toknes = ko_tokenizer.encode(args.prompt, return_tensors="pt")
    out_tokens = seq2seq.predict_seq(ko_toknes, en_tokenizer.encode('<bos>'), en_tokenizer.encode('<eos>'))
    result = en_tokenizer.decode(out_tokens.squeeze(1).detach().tolist())
    print(result)
    

def main(args):
    if args.task == 'gen':
        generate(args)
    elif args.task == 'trans':
        translate(args)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('gen.py')
    arg_parser.add_argument('--task', choices=['gen', 'trans'], default='gen')
    arg_parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    arg_parser.add_argument("--prompt", type=str, default='이것은 다시 말해')
    arg_parser.add_argument("--len", type=int, default=20)
    main(arg_parser.parse_args())