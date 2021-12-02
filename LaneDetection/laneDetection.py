from Image.image import Image
from LaneDetection.linearFunction import LinearFunction

from skimage import io, feature, transform, color, filters, draw, morphology, exposure, restoration
import matplotlib.pyplot as plt
import numpy as np


class LaneDetection(Image):
    def __init__(self, path: str, as_gray: bool = True):
        super().__init__(path, as_gray)
        self.path = path

    def isolateLane(self, trapezoid: np.array):
        """
        Isolate trapezoid from the image.
        :return: None
        """
        image = np.array(self.image, dtype=np.float64)
        height, width = self.image.shape

        mask = np.zeros(shape=(height, width))
        rows, columns = draw.polygon(trapezoid[:, 1], trapezoid[:, 0], (height, width))
        mask[rows, columns] = 1
        image *= mask
        return image


    def getMedian(self, points):
        x, y = [], []
        for point in points:
            x.append(point[0])
            y.append(point[1])
        try:
            a, b = np.polyfit(x, y, 1)
        except TypeError:
            return None

        line = LinearFunction(a, b)
        dev = [abs(y[i] - line.getValue(x[i])) for i in range(len(y))]
        for _ in range(int(len(y) * 0.2)):
            i = dev.index(max(dev))
            x.pop(i)
            y.pop(i)
            dev.pop(i)

        a, b = np.polyfit(x, y, 1)
        return LinearFunction(a, b)

    def findLane(self):
        """
        Detects lane
        :return: None
        """
        left = []
        right = []

        trapezoid = np.array([
            (0.2 * self.w, self.h * 0.9),
            (0.2 * self.w, self.h),
            (0.8 * self.w, self.h),
            (0.8 * self.w, self.h * 0.9),
            (int(self.w * 0.7), self.h * 0.75),
            (int(self.w * 0.3), self.h * 0.75),
        ])

        trapezoidLeft = np.array([
            (0, self.h),
            (0, 0.9 * self.h),
            (self.w * 0.3, self.h * 0.75),
            (self.w * 0.5, self.h * 0.75),
            (self.w * 0.5, self.h)
        ])

        trapezoidRight = np.array([
            (self.w * 0.5, self.h),
            (self.w * 0.5, self.h * 0.75),
            (self.w * 0.7, self.h * 0.75),
            (self.w, self.h * 0.9),
            (self.w, self.h)
        ])

        self.image = exposure.adjust_sigmoid(self.image)
        self.image = restoration.denoise_bilateral(self.image,
                                                   sigma_color=0.5,
                                                   sigma_spatial=0.1)

        medianValue = np.median(self.image)
        sigma = 0.001
        lowerBound = max(0.5, (1.0 - sigma) * medianValue)
        upperBound = min(1.0, (1.0 + sigma) * medianValue)

        self.image = feature.canny(self.image, sigma=sigma,
                                   low_threshold=lowerBound,
                                   high_threshold=upperBound)
        imgCopy = self.isolateLane(trapezoid)
        imgCopy = morphology.dilation(imgCopy, morphology.square(3))

        for i in range(10):
            points = transform.probabilistic_hough_line(imgCopy,
                                                        threshold=50,
                                                        line_length=20,
                                                        seed=i * 100)
            for point in points:
                x1, y1 = point[0]
                x2, y2 = point[1]
                try:
                    a = (y1 - y2) / (x1 - x2)
                    b = y1 - a * x1
                    line = LinearFunction(a, b)
                    if a <= -0.3 and (-0.2 * self.w <= line.getArgument(self.h) <= 0.4 * self.w):  # potential left border
                        left.append(point[0])
                        left.append(point[1])
                    elif a >= 0.3 and (0.6 * self.w <= line.getArgument(self.h) <= 1.1 * self.w):   # potential right border
                        right.append(point[0])
                        right.append(point[1])
                except ZeroDivisionError:
                    continue

        mini = self.getMedian(left)
        if mini is None:
            left = []
            imgCopy = self.isolateLane(trapezoidLeft)
            imgCopy = morphology.dilation(imgCopy, morphology.square(3))
            for _ in range(10):
                points = transform.probabilistic_hough_line(imgCopy,
                                                            threshold=50,
                                                            line_length=20,
                                                            seed=90)
                for point in points:
                    x1, y1 = point[0]
                    x2, y2 = point[1]
                    try:
                        a = (y1 - y2) / (x1 - x2)
                        b = y1 - a * x1
                        line = LinearFunction(a, b)
                        if a <= -0.2 and (-0.2 * self.w <= line.getArgument(self.h) <= 0.4 * self.w):  # potential left border
                            left.append(point[0])
                            left.append(point[1])
                    except ZeroDivisionError:
                        continue
            mini = self.getMedian(left)

        maxi = self.getMedian(right)
        if maxi is None:
            right = []
            imgCopy = self.isolateLane(trapezoidRight)
            imgCopy = morphology.dilation(imgCopy, morphology.square(3))
            for _ in range(10):
                points = transform.probabilistic_hough_line(imgCopy,
                                                            threshold=50,
                                                            line_length=20,
                                                            seed=90)
                for point in points:
                    x1, y1 = point[0]
                    x2, y2 = point[1]
                    try:
                        a = (y1 - y2) / (x1 - x2)
                        b = y1 - a * x1
                        line = LinearFunction(a, b)
                        if a >= 0.2 and (0.6 * self.w <= line.getArgument(self.h) <= 1.1 * self.w):  # potential right border
                            right.append(point[0])
                            right.append(point[1])
                    except ZeroDivisionError:
                        continue
            maxi = self.getMedian(right)

        return mini, maxi
        # plt.clf()
        # if mini is not None:
        #     plt.plot(
        #         [0, mini.getArgument(self.h * 0.85)],
        #         [mini.getValue(0), self.h * 0.85],
        #         color='red', lw=10, alpha=0.5
        #     )
        #
        # if maxi is not None:
        #     plt.plot(
        #         [maxi.getArgument(self.h * 0.85), self.w],
        #         [self.h * 0.85, maxi.getValue(self.w)],
        #         color='red', lw=10, alpha=0.5
        #     )
        #
        # plt.imshow(self.finalImage)
        # if path is None:
        #     plt.show()
        # else:
        #     plt.savefig(path, bbox_inches='tight', dpi=300)
        #     #io.imsave(path, self.finalImage)
        #     print(f"{path} saved")

