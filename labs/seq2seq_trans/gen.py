import argparse
import transformers
import model

LANG_CHOICE = set(["ko","en"])

def main(args):
    assert args.lang in LANG_CHOICE
    tokenizer = model.get_default_tokenizer(args.lang)
    encoder = model.KorEncoder.load_from_checkpoint(f'model/encoder/{args.lang}/model.ckpt')
    tokens = tokenizer.encode(args.prompt, return_tensors="pt")
    result = encoder.generate(tokens, args.len, tokenizer)
    print(result)
        

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('predict.py')
    arg_parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    arg_parser.add_argument("--prompt", type=str, default='이것은 다시 말해')
    arg_parser.add_argument("--len", type=int, default=20)
    main(arg_parser.parse_args())