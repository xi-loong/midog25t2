import os

import numpy as np

import cv2
import spams


def stain_mask(I, thresh=(0.1, 0.8)):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L > thresh[0]) & (L < thresh[1])


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def get_stain_matrix(I, threshold=(0.1, 0.8), lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = stain_mask(I, thresh=threshold).reshape((-1,))
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = normalize_rows(dictionary)
    return dictionary


###
def standardize_brightness(I):
    """
    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def get_concentrations(I, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T


class HeNormalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = np.array(
            [[0.65, 0.70, 0.29],
             [0.07, 0.99, 0.11]]
        )

    def __call__(self, I):
        I = standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)

import tqdm
if __name__ == '__main__':
    roots = [
        [r'E:\dataset\midog2025\midog25', r'E:\dataset\midog2025\midog25-stain'],
        [r'E:\dataset\midog2025\AtypicalMitoses', r'E:\dataset\midog2025\AtypicalMitoses-stain'],
        [r'E:\dataset\midog2025\AmiBr\atypical', r'E:\dataset\midog2025\AmiBr-stain\atypical'],
        [r'E:\dataset\midog2025\AmiBr\normal', r'E:\dataset\midog2025\AmiBr-stain\normal']
    ]
    normalizer = HeNormalizer()
    for input_root, output_root in roots:
        for filename in tqdm.tqdm(os.listdir(input_root)):
            if filename.endswith('.png'):
                stem, ext = os.path.splitext(filename)
                bgr = cv2.imread(f'{input_root}/{filename}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                he = cv2.cvtColor(normalizer(rgb), cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_root}/{filename}', he)
