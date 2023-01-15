import pytorch_lightning as pl
from torch.utils import data
from tqdm import tqdm
import torch


def sweep_kahead_pred(model: pl.LightningModule, dataset: data.Dataset, k_sweep, device: torch.DeviceObjType, batch_size:int=50):
    result = []
    for k_ahead in k_sweep:
        ffn_pred = predict_k_step_ahead(model, dataset, k_ahead, device=device, batch_size=batch_size)
        result.append(ffn_pred)
    return result
    


def predict_k_step_ahead(model: pl.LightningModule, dataset: data.Dataset, ahead_step: int = 5, device=torch.device("cpu"), batch_size:int=50):
    dataloader = data.DataLoader(dataset, batch_size=batch_size)
    preds = []
    for X, y in tqdm(iter(dataloader)):
        y = y.to(device)
        X = X.to(device)
        model = model.to(device)
        model.eval()
        y_hat = model.predict_k_step(X, ahead_step)
        preds.append(y_hat[:,-1,:])
    preds = torch.cat(preds)
    return preds.squeeze(1)
    