import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import syft as sy
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
from collections import defaultdict

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
dataloader = sy.FederatedDataLoader(dataset, shuffle = True, batch_size = 128)

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # print("got here")
        x = x.unsqueeze(1).float()
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

model = Net()
# model = torch.jit.script(AddingModel())
# model = torch.jit.trace(model, torch.Tensor([1, 2, 3]))
# model = torch.jit.trace(nn.Linear(3, 1), torch.Tensor([1.0, 2.0, 3.0]))
loss_fn = F.nll_loss

print("STARTING TO TRAIN")

def train():
    # Training Logic
    # opt = optim.SGD(params=model.parameters(), lr = .01)
    opt_dict = defaultdict(lambda: optim.Adam(params = model.parameters()))

    for i in range(20):

        total_loss = 0
        total_acc = 0
        n_iters = 0
        n_samples = 0

        for data, target in dataloader:
            # opt = optim.Adam(params = model.parameters())
            print(data.location)
            # print("sending the model")
            model.send(data.location)
            print("sent the model")
            print(data.location)
            output = model(data)
            print(output.location)
            # print(data.copy().get())
            # print(output.copy().get())
            # print(target.copy().get())
            print("got the output")
            # output.send(data.location)
            # target.send(data.location)
            # loss = ((output - target.squeeze())**2).sum()
            # print(loss)
            # model.get()
            loss = loss_fn(output, target.squeeze())
            values, classes = output.max(1)
            acc = (target == classes).sum() / float(data.shape[0])
            print(loss.location)
            print(target.location)
            # opt.zero_grad()
            opt_dict[data.location].zero_grad()
            print("zeroed the gradients")
            loss.backward()
            print("backpropogated")
            # opt.step()
            opt_dict[data.location].step()
            print("stepped optimizer")
            model.get()
            print("retrieved model")

            total_loss += loss.get()
            total_acc += acc.get()
            n_iters += 1
            n_samples += data.shape[0]
            print(n_samples)
            # data.location.remove_all_but_data() # do this because garbage collection has a problem
            # data.location._print_objects_on_remote()

        print(total_loss / n_iters)
        print(total_acc / n_iters)

train()



print("finished!")