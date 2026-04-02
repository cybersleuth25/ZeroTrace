"""Level 7: PyTorch Distributed Data Parallel (DDP) Batch Dimension Bug."""

DESCRIPTION = (
    "A researcher is setting up a basic PyTorch model using DistributedDataParallel (DDP). "
    "However, when running the forward pass, the model throws a RuntimeError. "
    "Identify why the model's Linear layer and DDP wrapper cannot process the input tensor, "
    "and apply the correct dimension mapping."
)

BUGGY_CODE = \
'''import torch
import torch.nn as nn
import torch.optim as optim
import os

# PyTorch Distributed Data Parallel Environment
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=0, world_size=1)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def train_step():
    setup()
    
    model = Network()
    ddp_model = DDP(model)
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Bug: DDP and nn.Linear require a batch dimension, but this is a 1D tensor shape (10,)
    inputs = torch.randn(10)
    
    # Forward pass crashes here
    outputs = ddp_model(inputs)
    
    loss = outputs.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    train_step()
'''

CORRECT_CODE = \
'''import torch
import torch.nn as nn
import torch.optim as optim
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=0, world_size=1)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def train_step():
    setup()
    
    model = Network()
    ddp_model = DDP(model)
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Fixed: Added a batch dimension to create shape (1, 10)
    inputs = torch.randn(1, 10)
    
    outputs = ddp_model(inputs)
    
    loss = outputs.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    train_step()
'''
