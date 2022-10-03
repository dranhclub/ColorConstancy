import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Kernel(nn.Module):
  def forward(self, x):
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]
    out = [r, g, b, r**2, g**2, b**2, r*g, g*b, r*b, r*g*b, torch.ones_like(r)]
    return torch.stack(out, dim=1)


class SCMN(nn.Module):
  def __init__(self, mode='train'):
    super().__init__()

    assert mode == 'train' or mode == 'eval'
    self.mode = mode

    self.Conv1 = ConvBlock(3, 24)
    self.Conv2 = ConvBlock(24, 48)
    self.Conv3 = ConvBlock(48, 96)
    self.Conv4 = ConvBlock(96, 192)
    self.Conv5 = ConvBlock(192, 384)
    self.MaxPool = nn.MaxPool2d(2, 2)
    self.Flatten = nn.Flatten()
    self.Linear = nn.Linear(24576, 33)
    self.Kernel = Kernel()

  def calc_mapping_matrix(self, x):
    x = self.MaxPool(self.Conv1(x))
    x = self.MaxPool(self.Conv2(x))
    x = self.MaxPool(self.Conv3(x))
    x = self.MaxPool(self.Conv4(x))
    x = self.MaxPool(self.Conv5(x))
    x = self.Flatten(x)
    x = self.Linear(x)
    x = torch.reshape(x, (-1, 3, 11))
    return x

  def forward(self, x):
    # img_size = (x.shape[2], x.shape[3])
    # assert img_size == (256, 256)

    mapping_matrix = self.calc_mapping_matrix(x)

    if self.mode == 'train':
      x = self.Kernel(x)
      x = torch.reshape(x, (-1, 11, 256 * 256))
      x = torch.bmm(mapping_matrix, x)
      x = torch.reshape(x, (-1, 3, 256, 256))
      return x
    else:
      return mapping_matrix