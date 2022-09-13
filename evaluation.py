import torch
import numpy as np
import os
from torchvision import transforms
from sklearn.linear_model import LinearRegression
from model import deepWBnet
import cv2
from evaluation.get_metadata import get_metadata
from evaluation.evaluate_cc import evaluate_cc
import sys
import logging
from datetime import datetime
# import matplotlib.pyplot as plt

################################# Setup logger #################################
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_handler = logging.FileHandler(filename=f'logs/evaluation_{timestamp}.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('LOGGER_NAME')

################################################################################

mytransform = transforms.Compose([
    transforms.Resize((128, 128))
])

net = deepWBnet()
net.load_state_dict(torch.load("checkpoints/model_20220912_123225_e10.pth"))
net.eval()

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))


def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I

def infer(image):
    """image: float np array RGB image"""
    with torch.no_grad():
        input = torch.from_numpy(image)
        input = input.permute((2,0,1))
        input = mytransform(input)
        image1 = input.permute((1,2,0)).numpy()
        input = input.to(torch.float32)
        input = input.unsqueeze(0)
        output = net(input)[0]
    output = output.permute((1,2,0)).numpy()
    mapping_func = get_mapping_func(image1, output)
    return outOfGamutClipping(apply_mapping_func(image, mapping_func))

def eval(name, imgin, gtimg):
    logger.info(f'RenderedWB_Set1/{name}')
    metadata_base = r'E:\datasets\WB\Set1\Set1_input_images_metadata'

    # read the image
    I_in = cv2.imread(imgin, cv2.IMREAD_COLOR)
    # read gt image
    gt = cv2.imread(gtimg, cv2.IMREAD_COLOR)
    # metadata
    metadata = get_metadata(name, 'RenderedWB_Set1', metadata_base)  

    # white balance I_in
    I_corr = (infer(I_in[:,:,::-1] / 255)[:,:,::-1]*255).astype("uint8")  

    # plt.figure(figsize=(20,10))
    # plt.subplot(1,3,1)
    # plt.imshow(I_in[:,:,::-1])
    # plt.title("Input")
    # plt.subplot(1,3,2)
    # plt.imshow(gt[:,:,::-1])
    # plt.title("Target")
    # plt.subplot(1,3,3)
    # plt.imshow(I_corr[:,:,::-1])
    # plt.title("Corrected")
    # plt.show()

    # Evaluation
    deltaE00, MSE, MAE, deltaE76 = evaluate_cc(I_corr, gt, metadata["cc_mask_area"],
                                            opt=4)
    logger.info('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n'
        % (deltaE00, MSE, MAE, deltaE76))

    return deltaE00, MSE, MAE, deltaE76

if __name__ == "__main__":
    img_dir = r'E:\datasets\WB\Set1\Set1_input_images_wo_CC_JPG'
    gt_dir = r'E:\datasets\WB\Set1\Set1_ground_truth_images_wo_CC'
    fold_file = r"E:\datasets\WB\Set1\Set1_folds\fold_1.txt"

    imgs = []    
    with open(fold_file, 'r') as f:
        imgs += [s.strip().split(".")[0] for s in f.readlines()]

    sum_deltaE00 = 0
    sum_MSE = 0 
    sum_MAE = 0
    sum_deltaE76 = 0

    n = 10
    for i in range(n):
        logger.info(f'{i+1}/{n}')
        img_in = os.path.join(img_dir, imgs[i] + '.jpg')
        img_gt = os.path.join(gt_dir, imgs[i].rsplit('_', maxsplit=2)[0] + '_G_AS.png')
        deltaE00, MSE, MAE, deltaE76 = eval(imgs[i], img_in, img_gt)
        sum_deltaE00 += deltaE00
        sum_MSE += MSE
        sum_MAE += MAE
        sum_deltaE76 += deltaE76
    
    avg_deltaE00 = sum_deltaE00 / n
    avg_MSE = sum_MSE / n
    avg_MAE = sum_MAE / n
    avg_deltaE76 = sum_deltaE76 / n

    logger.info('[AVG] DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n'
        % (avg_deltaE00, avg_MSE, avg_MAE, avg_deltaE76))