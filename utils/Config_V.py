import torch
from benchmarks.Exampler_V import Example


class CegisConfig():
    N_HIDDEN_NEURONS = [10]
    EXAMPLE = None
    ACTIVATION = ['SQUARE']
    BATCH_SIZE = 500
    LEARNING_RATE = 0.01
    LOSS_WEIGHT = (1, 1)
    BIAS = False
    SPLIT_D = False
    MARGIN = 0.5
    DEG = [2, 2]
    OPT = torch.optim.AdamW
    CHOICE = [0, 0]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
