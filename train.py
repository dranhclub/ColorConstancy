# %%
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
from sklearn.model_selection import train_test_split

# %% [markdown]
# # Dataset, dataloader

# %%
class MyDataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.img_dir = 'datasets/five_k_expert_a_small/images/'
        if train:
            self.label_dir = 'datasets/five_k_expert_a_small/train_data.json'
        else:
            self.label_dir = 'datasets/five_k_expert_a_small/test_data.json'
            
        with open(self.label_dir) as f:
            self.image_fn_and_illuminant = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.image_fn_and_illuminant)

    def __getitem__(self, index):
        image_fn, illuminant = self.image_fn_and_illuminant[index]
        
        # Read image
        path = os.path.join(self.img_dir, image_fn)
        image = read_image(path) / 255
        
        # Multiply image and illuminant
        illuminant = np.array(illuminant)
        image = np.multiply(image, illuminant[:, None, None])
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Convert to float32 (not use float64)
        return image.to(torch.float32), illuminant

# %%
my_transforms = transforms.Compose([
    transforms.Resize((128, 128))
])

train_dataloader = DataLoader(MyDataset(train=True, transform=my_transforms), 64, shuffle=True)
test_dataloader = DataLoader(MyDataset(train=False, transform=my_transforms), 64, shuffle=True)

# %%
# Show batch
plt.figure(figsize=(15, 15))
images, labels = next(iter(train_dataloader))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    image = torch.permute(images[i], (1, 2, 0))
    plt.imshow(image)
    plt.axis("off")

# %% [markdown]
# # Estimate-Color Network

# %%
class EstimateColorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.lastblock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*192, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.lastblock(x)
        return x

    

# %%
device = 'cuda'
model = EstimateColorNetwork().to(device)

# %% [markdown]
# # Criterion, optimizer

# %%
def loss_fn(outputs, targets):
    return torch.arccos(torch.cosine_similarity(outputs, targets)).mean()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# %% [markdown]
# # Train

# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter('runs/EstimateColor_{}'.format(timestamp))
n_epochs = 200

def train_one_epoch(epoch_index):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
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
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
    avg_loss = running_loss / (i + 1)
    return avg_loss

for epoch_index in range(n_epochs):
    print('EPOCH {}:'.format(epoch_index))

    model.train(True)
    train_loss = train_one_epoch(epoch_index)
    model.train(False)
    val_loss = validation()

    print('LOSS train {} valid {}'.format(train_loss, val_loss))
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : train_loss, 'Validation' : val_loss },
                    epoch_index + 1)
    writer.flush()


## Save model

model_path = 'checkpoints/model_{}.pth'.format(timestamp)
torch.save(model.state_dict(), model_path)
print("Saved " + model_path)
