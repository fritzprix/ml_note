from model import FFNAutoregressor, GRUAutoregress, SinusoidalTimeSeries
from argparse import Namespace, ArgumentParser
import pytorch_lightning as pl
from torch.utils import data
from tqdm import tqdm
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import utils
    

def main(args: Namespace):
    dataset = SinusoidalTimeSeries(0, n_samples=4000, noise=args.noise)
    
    ffn_model = FFNAutoregressor.load_from_checkpoint('./model/ffn/model.ckpt')
    gru_model = GRUAutoregress.load_from_checkpoint('./model/gru/model.ckpt')
    
    sweep_range = range(args.start, args.end, args.step)
    ref_range = range(0, 1)  # if step_ahead == 0, it's not prediction.
    sweep_range = [x for x in sweep_range if x not in ref_range]
    
    device = torch.device("cuda")
    ffn_result = utils.sweep_kahead_pred(ffn_model, dataset, sweep_range, device=device,batch_size=args.batch)
    gru_result = utils.sweep_kahead_pred(gru_model, dataset, sweep_range, device=device, batch_size=args.batch)
    ref = utils.predict_k_step_ahead(gru_model, dataset=dataset, ahead_step=0, device=device)
    ffn_loss = [F.mse_loss(result, ref) for result in ffn_result]
    gru_loss = [F.mse_loss(result, ref) for result in gru_result]
    
    
    if args.plot:
        plt.plot(ref.cpu().detach().numpy(), label="Ref.")
        for index, k in enumerate(sweep_range):
            plt.plot(ffn_result[index].cpu().detach().numpy(), label=f"ffn_step_ahead={k}_loss={ffn_loss[index]}", linestyle='-.')
            plt.plot(gru_result[index].cpu().detach().numpy(), label=f"gru_step_ahead={k}_loss={gru_loss[index]}", linestyle=':')
        plt.legend()
        plt.show()
    


if __name__ == '__main__':
    arg_parser = ArgumentParser("test.py")
    arg_parser.add_argument("--device", type=str, choices=["cpu","gpu"], default="gpu")
    arg_parser.add_argument("--start", type=int, default=0)
    arg_parser.add_argument("--end", type=int, default=500)
    arg_parser.add_argument("--step", type=int, default=100)
    arg_parser.add_argument("--plot", type=bool, default=True)
    arg_parser.add_argument("--batch", type=int, default=50)
    arg_parser.add_argument("--noise", type=float, default=0)
    main(arg_parser.parse_args())
    
