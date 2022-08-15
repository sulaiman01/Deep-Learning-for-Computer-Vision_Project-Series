import numpy as np
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super.__init__()
        self.layer1 = nn.Conv2d()