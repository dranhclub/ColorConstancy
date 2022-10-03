import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from torch import optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset import RendererWB
from model import SCMN

class Trainer():
    def __init__(self, input_img_dir, target_img_dir, checkpoints_dir, folds_dir):
        # Dataloaders
        self.input_img_dir = input_img_dir
        self.target_img_dir = target_img_dir
        self.folds_dir = folds_dir
        self.train_dataloader, self.test_dataloader = self._get_dataloaders()
        
        # Net
        self.device = 'cuda'
        self.net = SCMN(mode='train').to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 25, gamma=0.5, last_epoch=-1)

        # Logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/SCMN_{self.timestamp}')

        # Make checkpoint directory
        self.checkpoints_dir = checkpoints_dir
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
            print("Makedirs ", self.checkpoints_dir)

    def _train_one_epoch(self, epoch_index):
        running_loss = 0.0
        n_iters = len(self.train_dataloader)
        
        for i, data in enumerate(self.train_dataloader):
            inputs, targets = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self._loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print(f'  batch [{i+1}/{n_iters}] loss: {last_loss}')
                tb_x = epoch_index * n_iters + i + 1
                self.writer.add_scalars('Loss', {"Train": last_loss}, tb_x)
                running_loss = 0.

        self.writer.flush()
        return last_loss

    def _validate(self):
        print("Validating")
        running_loss = 0.0
        for i, data in enumerate(self.test_dataloader):
            inputs, targets = data[0].to(self.device), data[1].to(self.device)
            outputs = self.net(inputs)
            loss = self._loss_fn(outputs, targets)
            running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        return avg_loss

    def _loss_fn(self, output, target):
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss

    def _get_dataloaders(self):
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(0.5)
        ])

        train_dataset = RendererWB(self.input_img_dir, self.target_img_dir, self.folds_dir, train=True, transform=transform)
        test_dataset = RendererWB(self.input_img_dir, self.target_img_dir, self.folds_dir, train=False, transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
        test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8)

        return train_dataloader, test_dataloader

    def train(self, epochs=150, model_name='scmn', checkpoints_dir='./checkpoints'):
        for epoch_index in range(epochs):
            print(f'EPOCH {epoch_index + 1}:')

            self.net.train(True)
            train_loss = self._train_one_epoch(epoch_index)
            self.net.train(False)
            val_loss = self._validate()

            self.scheduler.step()

            print(f'LOSS train {train_loss} valid {val_loss}')
            self.writer.add_scalars('Loss', {'Valid' : val_loss }, (epoch_index + 1) * len(self.train_dataloader))
            self.writer.flush()

            if epoch_index % 5 == 4:
                model_path = os.path.join(checkpoints_dir, f'{model_name}_{self.timestamp}_{epoch_index+1}.pth')
                torch.save(self.net.state_dict(), model_path)
                print("Saved " + model_path)
        
        model_path = os.path.join(checkpoints_dir, f'{model_name}_{self.timestamp}_done.pth')
        torch.save(self.net.state_dict(), model_path)
        print("Saved " + model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Simple Color Mapping Network.")
    parser.add_argument('--checkpoints-dir', default="./checkpoints")
    parser.add_argument('--input-img-dir', default='E:/datasets/WB/Set1/Set1_input_images_wo_CC_JPG')
    parser.add_argument('--target-img-dir', default='E:/datasets/WB/Set1/Set1_ground_truth_images_wo_CC')
    parser.add_argument('--folds-dir', default='E:/datasets/WB/Set1/Set1_folds')
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()

    print(args)
    print("Training...")

    trainer = Trainer(
        args.input_img_dir,
        args.target_img_dir,
        args.checkpoints_dir,
        args.folds_dir
    )

    trainer.train(args.epochs)
