from model import SCMN
import torch
from util import outOfGamutClipping
from torchvision import transforms
import numpy as np
from imageio import imread

class ColorCorrector():
    def __init__(self, model_checkpoint_path):
        self.net = SCMN(mode='eval')
        self.net.load_state_dict(torch.load(model_checkpoint_path))
        self.net.eval()
        if torch.cuda.is_available():
            self.net = self.net.to('cuda')

        self.transform =  transforms.Compose([
            transforms.Resize((256, 256))
        ])
    
    def get_mapping_matrix(self, img):
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.permute(img, (2, 0, 1))
        img = torch.unsqueeze(img, dim=0)
        img = img.to('cuda')
        with torch.no_grad():
            img = self.transform(img)
            out = self.net(img)  # 1 x 3 x 11
        out = torch.squeeze(out)  # 3 x 11
        mapping_matrix = out.cpu().numpy()
        return mapping_matrix


    def kernel(self, img):
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        # 11 x height x width
        return np.stack([r, g, b, r**2, g**2, b**2, r*g, g*b, r*b, r*g*b, np.ones_like(r)], axis=0)


    def correct(self, img):
        M = self.get_mapping_matrix(img)  # 3 x 11
        height, width, _ = img.shape
        img = self.kernel(img)  # 11 x height x width
        img = img.reshape(11, -1)  # (11 x width*height)
        img = np.matmul(M, img)  # 3 x width * height
        img = img.reshape(3, height, width)  # 3 x height x width
        img = np.transpose(img, (1, 2, 0))  # height x width x 3
        img = outOfGamutClipping(img)
        return img

    def correct_img_with_path(self, img_path):
        img = imread(img_path) / 255
        return self.correct(img)