# %%
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torch.nn as nn
from torch import optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# %% [markdown]
# # Data

# %%
class RendererWB(Dataset):
    def __init__(self, train=True, transform=None):
        self.img_dir = '/media/HDD2/hoangnh/CC/datasets/WB/Set1_input_images_wo_CC_JPG'
        self.gt_dir = '/media/HDD2/hoangnh/CC/datasets/WB/Set1_ground_truth_images_wo_CC'
        fold_base = '/media/HDD2/hoangnh/CC/datasets/WB/Set1_folds'

        # Get imgs from folds
        if train == True:
            folds = ['fold_2.txt', 'fold_3.txt']
        else:
            folds = ['fold_1.txt']

        self.imgs = []
        for fold in folds:
            with open(os.path.join(fold_base, fold), 'r') as f:
                self.imgs += [s.strip().split(".")[0] for s in f.readlines()]

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # Get file names
        img_fn = self.imgs[index]
        in_img_fn = os.path.join(self.img_dir, img_fn + '.jpg')
        gt_img_fn = os.path.join(self.gt_dir, img_fn.rsplit('_', maxsplit=2)[0] + '_G_AS.png')

        # Read img
        in_img = read_image(in_img_fn) / 255
        gt_img = read_image(gt_img_fn) / 255
        
        # Transform
        in_img = self.transform(in_img)
        gt_img = self.transform(gt_img)

        return in_img.to(torch.float32), gt_img.to(torch.float32)

mytransform = transforms.Compose([
    transforms.Resize((128, 128))
])

train_dataset = RendererWB(train=True, transform=mytransform)
test_dataset = RendererWB(train=False, transform=mytransform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# %% [markdown]
# # Model

# %%
class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_up(x)



class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)



    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)


class deepWBnet(nn.Module):
    def __init__(self):
        super(deepWBnet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = UpBlock(192, 96)
        self.decoder_up2 = UpBlock(96, 48)
        self.decoder_up3 = UpBlock(48, 24)
        self.decoder_out = OutputBlock(24, self.n_channels)


    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x = self.decoder_up3(x, x2)
        out = self.decoder_out(x, x1)
        return out


def loss_fn(output, target):
    loss = torch.sum(torch.abs(output - target)) / output.size(0)
    return loss



net = deepWBnet()

lrdf=0.5
lrdp=25


optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)

# %% [markdown]
# # Train

# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f'runs/Generative_{timestamp}')
device = 'cuda'

net = net.to(device)

def train_one_epoch(epoch_index):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if torch.isnan(outputs).any():
            raise Exception("Output is NaN")
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    writer.flush()
    return last_loss

def validation():
    running_loss = 0.0
    for i, data in enumerate(test_dataloader):
        inputs, targets = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
        running_loss += loss.item()
    avg_loss = running_loss / (i + 1)
    return avg_loss

# %%
## Run train
for epoch_index in range(110):
    print('EPOCH {}:'.format(epoch_index))

    net.train(True)
    train_loss = train_one_epoch(epoch_index)
    net.train(False)
    val_loss = validation()

    print('LOSS train {} valid {}'.format(train_loss, val_loss))
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : train_loss, 'Validation' : val_loss },
                    epoch_index + 1)
    writer.flush()


# %%
model_path = 'checkpoints/model_{}.pth'.format(timestamp)
torch.save(net.state_dict(), model_path)
print("Saved " + model_path)

