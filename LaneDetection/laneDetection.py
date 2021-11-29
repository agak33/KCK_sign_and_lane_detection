from Image.image import Image
from LaneDetection.linearFunction import LinearFunction

from skimage import io, feature, transform, color, filters, draw, morphology, exposure
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List


class LaneDetection(Image):
    def __init__(self, path: str, as_gray: bool = True):
        super().__init__(path, as_gray)
        self.laneLines: List = []
        self.path = path

    def isolateLane(self):
        self.image = np.array(self.image, dtype=np.float64)
        height, width = self.image.shape
        trapezoid = np.array([
            (width // 10     , height),
            (width           , height),
            (int(width * 0.6), height // 2),
            (int(width * 0.3), height // 2),
        ])

        mask = np.zeros(shape=(height, width))
        rows, columns = draw.polygon(trapezoid[:, 1], trapezoid[:, 0], (height, width))
        mask[rows, columns] = 1
        self.image *= mask

    def laneFilter(self,
                   minValue: Union[float, tuple] = 0.7,
                   maxValue: Union[float, tuple] = 1):
        h, w = self.image.shape
        for i in range(h):
            for j in range(w):
                if maxValue >= self.image[i, j] >= minValue:
                    self.image[i, j] = 1
                else:
                    self.image[i, j] = 0

    def findLane(self):
        self.isolateLane()
        #self.laneFilter()

        self.image = feature.canny(self.image)
        self.show()

        # self.image = morphology.dilation(self.image, morphology.square(3))
        #
        # angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        # h, theta, d = transform.hough_line(self.image, theta=angles)
        #
        # for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d)):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     self.laneLines.append(
        #         LinearFunction(
        #             np.tan(angle + np.pi / 2),
        #             y0 - np.tan(angle + np.pi / 2) * x0,
        #             x0, y0
        #         )
        #     )
        #
        # h, w = self.image.shape
        #
        # mini = min(self.laneLines, key= lambda x: x.a)
        # maxi = max(self.laneLines, key= lambda x: x.a)
        #
        # plt.plot(
        #     [0, mini.getArgument(h // 1.4)],
        #     [mini.getValue(0), h // 1.4],
        #     color='red', lw=10, alpha=0.5
        # )
        # plt.plot(
        #     [maxi.getArgument(h // 1.4), w],
        #     [h // 1.4, maxi.getValue(w)],
        #     color='red', lw=10, alpha=0.5
        # )
        #
        # plt.imshow(self.finalImage)
        # plt.show()
