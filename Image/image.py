from skimage import io, feature, transform, color, filters, draw, morphology, exposure
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


class Image(object):
    def __init__(self, path: str, as_gray: bool = True):
        self.image = io.imread(path, as_gray=as_gray)
        self.finalImage = io.imread(path, as_gray=False)

    def show(self):
        plt.imshow(self.image)
        plt.show()


