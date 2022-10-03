import cv2
import numpy as np
from skimage import color
import os
import numpy as np
import re

###########################################################################################
# These evaluate functions is from 
# https://github.com/mahmoudnafifi/WB_sRGB/tree/master/WB_sRGB_Python/evaluation 


def calc_deltaE(source, target, color_chart_area):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    source = color.rgb2lab(source)
    target = color.rgb2lab(target)
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    delta_e = np.sqrt(np.sum(np.power(source - target, 2), 1))
    return sum(delta_e) / (np.shape(delta_e)[0] - color_chart_area)


def calc_deltaE2000(source, target, color_chart_area):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    source = color.rgb2lab(source)
    target = color.rgb2lab(target)
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    deltaE00 = deltaE2000(source, target)
    return sum(deltaE00) / (np.shape(deltaE00)[0] - color_chart_area)


def deltaE2000(Labstd, Labsample):
    kl = 1
    kc = 1
    kh = 1
    Lstd = np.transpose(Labstd[:, 0])
    astd = np.transpose(Labstd[:, 1])
    bstd = np.transpose(Labstd[:, 2])
    Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
    Lsample = np.transpose(Labsample[:, 0])
    asample = np.transpose(Labsample[:, 1])
    bsample = np.transpose(Labsample[:, 2])
    Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
    Cabarithmean = (Cabstd + Cabsample) / 2
    G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
        Cabarithmean, 7) + np.power(25, 7))))
    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
    Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
    Cpprod = (Cpsample * Cpstd)
    zcidx = np.argwhere(Cpprod == 0)
    hpstd = np.arctan2(bstd, apstd)
    hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
    hpsample = np.arctan2(bsample, apsample)
    hpsample = hpsample + 2 * np.pi * (hpsample < 0)
    hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
    dL = (Lsample - Lstd)
    dC = (Cpsample - Cpstd)
    dhp = (hpsample - hpstd)
    dhp = dhp - 2 * np.pi * (dhp > np.pi)
    dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
    dhp[zcidx] = 0
    dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
    Lp = (Lsample + Lstd) / 2
    Cp = (Cpstd + Cpsample) / 2
    hp = (hpstd + hpsample) / 2
    hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
    hp = hp + (hp < 0) * 2 * np.pi
    hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
    Lpm502 = np.power((Lp - 50), 2)
    Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
    Sc = 1 + 0.045 * Cp
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
        0.32 * np.cos(3 * hp + np.pi / 30) \
        - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
    Sh = 1 + 0.015 * Cp * T
    delthetarad = (30 * np.pi / 180) * np.exp(
        - np.power((180 / np.pi * hp - 275) / 25, 2))
    Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
    RT = - np.sin(2 * delthetarad) * Rc
    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh
    de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                    np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))
    return de00


def calc_mae(source, target, color_chart_area):
    source = np.reshape(source, [-1, 3]).astype(np.float32)
    target = np.reshape(target, [-1, 3]).astype(np.float32)
    source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
    target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
    norm = source_norm * target_norm
    L = np.shape(norm)[0]
    inds = norm != 0
    angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
    angles[angles > 1] = 1
    f = np.arccos(angles)
    f[np.isnan(f)] = 0
    f = f * 180 / np.pi
    return sum(f) / (L - color_chart_area)


def calc_mse(source, target, color_chart_area):
    source = np.reshape(source, [-1, 1]).astype(np.float64)
    target = np.reshape(target, [-1, 1]).astype(np.float64)
    mse = sum(np.power((source - target), 2))
    return mse / ((np.shape(source)[
        0]) - color_chart_area)


def evaluate_cc(corrected, gt, color_chart_area, opt=1):
    """
        Color constancy (white-balance correction) evaluation of a given corrected
        image.
        :param corrected: corrected image
        :param gt: ground-truth image
        :param color_chart_area: If there is a color chart in the image, that is
        masked out from both images, this variable represents the number of pixels
        of the color chart.
        :param opt: determines the required error metric(s) to be reported.
            Options:
            opt = 1 delta E 2000 (default).
            opt = 2 delta E 2000 and mean squared error (MSE)
            opt = 3 delta E 2000, MSE, and mean angular eror (MAE)
            opt = 4 delta E 2000, MSE, MAE, and delta E 76
        :return: error(s) between corrected and gt images
        """

    if opt == 1:
        return calc_deltaE2000(corrected, gt, color_chart_area)
    elif opt == 2:
        return calc_deltaE2000(corrected, gt, color_chart_area), calc_mse(
            corrected, gt, color_chart_area)
    elif opt == 3:
        return calc_deltaE2000(corrected, gt, color_chart_area), calc_mse(
            corrected, gt, color_chart_area), calc_mae(corrected, gt,
                                                    color_chart_area)
    elif opt == 4:
        return calc_deltaE2000(corrected, gt, color_chart_area), calc_mse(
            corrected, gt, color_chart_area), calc_mae(
            corrected, gt, color_chart_area), calc_deltaE(corrected, gt,
                                                        color_chart_area)
    else:
        raise Exception('Error in evaluate_cc function')


def get_metadata(fileName, set, metadata_baseDir=''):
    """
    Gets metadata (e.g., ground-truth file name, chart coordinates and area).
    :param fileName: input filename
    :param set: which dataset?--options includes: 'RenderedWB_Set1',
      'RenderedWB_Set2', 'Rendered_Cube+'
    :param metadata_baseDir: metadata directory (required for Set1 only)
    :return: metadata for a given image
    evaluation_examples.py provides some examples of how to use it
    """

    fname, file_extension = os.path.splitext(fileName)  # get file parts
    name = os.path.basename(fname)  # get only filename without the directory

    if set == 'RenderedWB_Set1':  # Rendered WB dataset (Set1)
        metadatafile_color = name + '_color.txt'  # chart's colors info.
        metadatafile_mask = name + '_mask.txt'  # chart's coordinate info.
        # get color info.
        f = open(os.path.join(metadata_baseDir, metadatafile_color), 'r')
        C = f.read()
        colors = np.zeros((3, 24))  # color chart colors
        temp = re.split(',|\n', C)
        # 3 x 24 colors in the color chart
        colors = np.reshape(np.asfarray(temp[:-1], float), (24, 3)).transpose()
        # get coordinate info
        f = open(os.path.join(metadata_baseDir, metadatafile_mask), 'r')
        C = f.read()
        temp = re.split(',|\n', C)
        # take only the first 4 elements (i.e., the color chart coordinates)
        temp = temp[0:4]
        mask = np.asfarray(temp, float)  # color chart mask coordinates
        # get ground-truth file name
        seperator = '_'
        temp = name.split(seperator)
        gt_file = seperator.join(temp[:-2])
        gt_file = gt_file + '_G_AS.png'
        # compute mask area
        mask_area = mask[2] * mask[3]
        # final metadata
        data = {"gt_filename": gt_file, "cc_colors": colors, "cc_mask": mask,
                "cc_mask_area": mask_area}

    elif set == 'RenderedWB_Set2':  # Rendered WB dataset (Set2)
        data = {"gt_filename": name + file_extension, "cc_colors": None,
                "cc_mask": None, "cc_mask_area": 0}

    elif set == 'Rendered_Cube+':  # Rendered Cube+
        # get ground-truth filename
        temp = name.split('_')
        gt_file = temp[0] + file_extension
        mask_area = 58373  # calibration obj's area is fixed over all images
        data = {"gt_filename": gt_file, "cc_colors": None, "cc_mask": None,
                "cc_mask_area": mask_area}
    else:
        raise Exception(
            "Invalid value for set variable. " +
            "Please use: 'RenderedWB_Set1', 'RenderedWB_Set2', 'Rendered_Cube+'")

    return data


def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I
