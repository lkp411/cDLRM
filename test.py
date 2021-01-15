import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

    def forward(self, y):
        return self.x * y


if __name__ == '__main__':
    net = TestNet()
    base_device = torch.device('cuda:0')
    # net.to(base_device)
    a = torch.randn(10, 10, device=base_device)
    b = 5
    net.x = 10
    print(net(b))
    # val = net(a, b)
    # print(val)


