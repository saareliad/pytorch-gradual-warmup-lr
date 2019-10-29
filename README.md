# pytorch-gradual-warmup-lr

Gradually warm-up(increasing) learning rate for pytorch's optimizer. Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

<img src="asset/tensorboard.png" alt="example tensorboard" width="300" height="whatever">
Example : Gradual Warmup for 100 epoch, after that, use cosine-annealing.

## Install

```
$ pip install git+https://github.com/saareliad/pytorch-gradual-warmup-lr.git
```

## Usage

```python
from warmup_scheduler import GradualWarmupScheduler

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)

for epoch in range(train_epoch):
    # train(...)
    # validate(...)
    scheduler_warmup.step()     # 10 epoch warmup, after that schedule as scheduler_cosine
    ...
```
## Changes from original repo
This is my own fork of https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
My changes add compatibility with Pytorch >= 1.1,
See discussion [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
(Kept on my own fork to maintain stability across existing projects)
