from LaneDetection.laneDetection import LaneDetection
from signDetection.signDetection.signDetectionC import  Detect
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    for i in range(66):
        try:
            img = LaneDetection(f'img/road{i}.jpg')
            mini, maxi = img.findLane()
            img1 = Detect(f'img/road{i}.jpg')
            img1.detectSign(i)



            plt.clf()

            if mini is not None:
                plt.plot(
                    [0, mini.getArgument(img1.h * 0.85)],
                    [mini.getValue(0), img1.h * 0.85],
                    color='red', lw=10, alpha=0.5
                )

            if maxi is not None:
                plt.plot(
                    [maxi.getArgument(img1.h * 0.85), img1.w],
                    [img1.h * 0.85, maxi.getValue(img1.w)],
                    color='red', lw=10, alpha=0.5
                )
            plt.imshow(img1.finalImage)

            plt.savefig(f'results/road{i}.jpg', bbox_inches='tight', dpi=300)
            print(f"f'results/road{i}.jpg' saved")
        except FileNotFoundError:
            continue




