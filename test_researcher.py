import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import syft as sy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys

from researcher_worker import ResearcherWorker
from researcher_dataloader import ResearcherDataset

PROXY_URL = "127.0.0.1"
PROXY_PORT = 8888
VERBOSE = True

hook = sy.TorchHook(torch)

# # researcher = ResearcherWorker(hook, PROXY_URL, PROXY_PORT, verbose = VERBOSE, id = "researcher" if len(sys.argv) == 1 else sys.argv[-1], is_client_worker = True) # try is_client_worker
# print(researcher.id)

# x = torch.Tensor([1, 2, 3, 4, 5])
# x_ptr = x.send(researcher)

# print(x_ptr)
# print(researcher)
# y = x_ptr + x_ptr
# print(y)
# z = y.get()
# print(z)

# # help(researcher)
# print(researcher.list_objects_remote())
# print(type(researcher.list_objects_remote()))

# print(type(researcher))

# print(researcher.search("data"))

dataset = ResearcherDataset(['santa', 'grinch'] if len(sys.argv) == 1 else sys.argv[1:])
dataloader = sy.FederatedDataLoader(dataset, shuffle = True)

# model = nn.Linear(2,1)

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(2, 3)
        self.output = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = torch.sigmoid(self.output(x))

        return x

class AddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(3, 1)

    # @torch.jit.script_method
    def forward(self, x):
        out = self.lin(x)

        return out

model = AddingModel()
# model = torch.jit.script(AddingModel())
# model = torch.jit.trace(model, torch.Tensor([1, 2, 3]))
# model = torch.jit.trace(nn.Linear(3, 1), torch.Tensor([1.0, 2.0, 3.0]))
loss_fn = nn.MSELoss()

print("STARTING TO TRAIN")

def train():
    # Training Logic
    opt = optim.Adam(params=model.parameters())
    for i in range(20):

        for data, target in dataloader:
            print(data.location)
            print("sending the model")
            model.send(data.location)
            print("sent the model")
            print("zeroed the gradients")
            output = model(data)
            print("got the output")
            loss = ((output - target.squeeze())**2).sum()
            # loss = -loss_fn(output, target.squeeze())
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.get()

            # # 1) erase previous gradients (if they exist)
            # opt.zero_grad()

            # # 2) make a prediction
            # pred = model(data)

            # # 3) calculate how much we missed
            # loss = ((pred - target)**2).sum()

            # # 4) figure out which weights caused us to miss
            # loss.backward()

            # # 5) change those weights
            # opt.step()

            # # 6) print our progress
        print(loss.get())

train()



print("finished!")