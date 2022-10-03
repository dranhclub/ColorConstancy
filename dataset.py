from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import torch

class RendererWB(Dataset):
    def __init__(self, in_dir, gt_dir, fold_dir, train=True, transform=None):
        self.img_dir = in_dir
        self.gt_dir = gt_dir
        fold_base = fold_dir

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
        combine_img = torch.cat((in_img, gt_img), dim=0)
        combine_img = self.transform(combine_img)
        in_img = combine_img[:3,...]
        gt_img = combine_img[3:,...]

        return in_img.to(torch.float32), gt_img.to(torch.float32)
