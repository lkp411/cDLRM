import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from timeit import default_timer as timer


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.layer = nn.Linear(10, 10)
        self.local = torch.ones(10, 10)

    def forward(self, input):
        return self.layer(input)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, idxs):
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.idxs[idx]



def example(rank, world_size, a , b, c):
    # create default process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model

    dataset = TestDataset(list(range(1000)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    model = TestModule()
    model.to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print(a, b, c)

    for i, idx in enumerate(dataloader):
        print(rank, idx)
        input = torch.randn(20, 10)
        ddp_model.module.local[0, :5] = 2 * rank
        outputs = ddp_model(input[rank * 10: (rank + 1) * 10, :].to(rank))
        labels = torch.randn(10, 10).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()


def main():
    world_size = 2
    mp.spawn(example,
             args=(world_size, 10, 11, 12),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
