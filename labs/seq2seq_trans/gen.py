import argparse
import model
import data
import torch

LANG_CHOICE = set(["ko","en"])

def main(args):
    assert args.lang in LANG_CHOICE
    tokenizer = data.get_default_tokenizer(args.lang)
    encoder = model.GRUModule.load_from_checkpoint(f'model/{args.lang}/model.ckpt')
    cuda = torch.device('cuda')
    tokens = tokenizer.encode(args.prompt, return_tensors="pt").to(cuda)
    encoder = encoder.cuda()
    print(encoder.generate(tokens, args.len, tokenizer))
        

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('gen.py')
    arg_parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    arg_parser.add_argument("--prompt", type=str, default='이것은 다시 말해')
    arg_parser.add_argument("--len", type=int, default=20)
    main(arg_parser.parse_args())