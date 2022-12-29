# CNN model experiment for classification task

## Tools
- ML framework : [pytorch]() & [Lightning]()
- Visualization : [wandb]()

## Performance (Tested on FashionMNIST Test Dataset)
|Name|Avg. Accuracy| 
|:-:|:-:|
|simple|91.9%|


## Running Environment
- WSL2 @ Windows 11
- [Python environment](./conda.env)

## Try out
> To see benchmark result for model

```bash
$ python simple.py --bm=True
```

## TO-DO
- [x]: Simple CNN model for classification task
- [ ]: ResNet (frozen) based classifier for FashionMNIST
