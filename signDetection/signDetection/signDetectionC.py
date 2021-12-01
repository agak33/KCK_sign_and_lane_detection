import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math
from Image.image import Image
from keras.models import load_model
import warnings


model = load_model(f'signDetection/Trafic_signs_model.h5')
warnings.filterwarnings("ignore", category=DeprecationWarning)
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing veh over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Veh > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing veh > 3.5 tons',
           44: 'Other'}


class detect(Image):
    def __init__(self, path: str, as_gray: bool = True):
        super().__init__(path)
        self.path = path


    ### Preprocess image
    def constrastLimit(self, ima):
        img_hist_equalized = cv2.cvtColor(ima, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(img_hist_equalized)
        img_hist_equalized[:,:,0] = cv2.equalizeHist(img_hist_equalized[:,:,0])
        img_hist_equalized = cv2.merge(channels)
        img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
        return img_hist_equalized


    def LaplacianOfGaussian(self, ima):
        LoG_image = cv2.GaussianBlur(ima, (3, 3), 0)  # paramter
        gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
        LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)  # parameter
        LoG_image = cv2.convertScaleAbs(LoG_image)
        return LoG_image


    def binarization(self, ima):
        thresh = cv2.threshold(ima, 32, 255, cv2.THRESH_BINARY)[1]
        return thresh


    def preprocess_image(self, ima):
        ima = self.constrastLimit(ima)
        ima = self.LaplacianOfGaussian(ima)
        ima = self.binarization(ima)
        return ima


    # Find Signs
    def removeSmallComponents(self, ima, threshold):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(ima, connectivity=8)
        sizes = stats[1:, -1];
        nb_components = nb_components - 1
        img2 = np.zeros((output.shape), dtype=np.uint8)
        # for every component in the image, you keep it only if it's above threshold
        for i in range(0, nb_components):
            if sizes[i] >= threshold:
                img2[output == i + 1] = 255
        return img2


    def findContour(self, ima):
        # find contours in the thresholded image
        cnts = cv2.findContours(ima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts2 = cv2.findContours(ima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts2 = imutils.grab_contours(cnts2)
        return cnts, cnts2


    def Shape(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        if peri > 250 :
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape = "triangle"
            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            else:
                shape = "circle"
            # return the name of the shape
        return shape


    def contourIsSign(self, perimeter, centroid, threshold):
        # # Compute signature of contour
        result = []
        for p in perimeter:
            p = p[0]
            distance = sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
            result.append(distance)
        max_value = max(result)
        signature = [float(dist) / max_value for dist in result]
        # Check signature of contour.
        temp = sum((1 - s) for s in signature)
        temp = temp / len(signature)
        if temp < threshold:  # is  the sign
            return True, max_value + 2
        else:  # is not the sign
            return False, max_value + 2


    # crop sign
    def cropSign(self, ima, coordinate):
        width = ima.shape[1]
        height = ima.shape[0]
        top = max([int(coordinate[0][1]), 0])
        bottom = min([int(coordinate[1][1]), height - 1])
        left = max([int(coordinate[0][0]), 0])
        right = min([int(coordinate[1][0]), width - 1])
        return ima[top:bottom, left:right]


    def findLargestSign(self, ima, contours, threshold, distance_theshold):
        max_distance = 0
        coordinate = None
        sign = None
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            is_sign, distance = self.contourIsSign(c, [cX, cY], 1 - threshold)
            if is_sign and distance > max_distance and distance > distance_theshold :
                max_distance = distance
                coordinate = np.reshape(c, [-1, 2])
                left, top = np.amin(coordinate, axis=0)
                right, bottom = np.amax(coordinate, axis=0)
                coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
        if coordinate is not None:
            y = list(coordinate[0])
            x = list(coordinate[1])
            przekatna = sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)
            y = [int(element - 0.1 * przekatna) for element in y]
            coordinate[0] = tuple(y)
            x = [int(element + 0.1 * przekatna) for element in x]
            coordinate[1] = tuple(x)
            sign = self.cropSign(ima, coordinate)
        return sign, coordinate


    def findOtherSigns(self, ima, contours):
        coordinate = None
        sign = None
        for c in contours:
            nS = ["triangle", "square"]
            nameShape = self.Shape(c)
            if nameShape in nS:
                coordinate = np.reshape(c, [-1, 2])
                left, top = np.amin(coordinate, axis=0)
                right, bottom = np.amax(coordinate, axis=0)
                coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
                break
        y = list(coordinate[0])
        x = list(coordinate[1])
        przekatna = sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)
        y = [int(element - 0.1 * przekatna) for element in y]
        coordinate[0] = tuple(y)
        x = [int(element + 0.1 * przekatna) for element in x]
        coordinate[1] = tuple(x)
        sign = self.cropSign(ima, coordinate)
        return sign, coordinate

    #classify type of sign
    def classify(self, ii):
        ii = cv2.resize(ii, (30, 30))
        ii = np.expand_dims(ii, axis=0)
        ii = np.array(ii)
        predict_x = model.predict(ii)
        classes_x = np.argmax(predict_x, axis=1)
        return classes_x


    def localization(self,ima, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
        original_image = ima.copy()
        binary_image = self.preprocess_image(ima)

        binary_image = self.removeSmallComponents(binary_image, min_size_components)

        binary_image = cv2.bitwise_and(binary_image, binary_image, mask=self.remove_other_color(ima))

        cv2.imshow('BINARY IMAGE', binary_image)
        #k = cv2.waitKey(0);

        contours, contursOther = self.findContour(binary_image)

        sign, coordinate = self.findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
        if sign is None and coordinate is None:
            sign, coordinate = self.findOtherSigns(original_image, contursOther)
        text = ""
        sign_type = -1
        i = 0
        sign_type = int(self.classify(sign)) + 1
        if sign is not None:
            sign_type = sign_type if sign_type <= 43 else 44
            text = classes[sign_type]
            cv2.imwrite(str(count) + '_' + text + '.jpg', sign)
        if sign_type > 0 :
            cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 4)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] + 15), font, 2, (0, 0, 255), 2,
                        cv2.LINE_4)
       #  return coordinate, original_image
        return coordinate, original_image, sign_type, text


    def remove_other_color(self, img):
        frame = cv2.GaussianBlur(img, (3, 3), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([100, 128, 0])
        upper_blue = np.array([215, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        lower_white = np.array([0, 0, 128], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        # Threshold the HSV image to get only blue colors
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([170, 150, 50], dtype=np.uint8)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_1 = cv2.bitwise_or(mask_blue, mask_white)
        mask = cv2.bitwise_or(mask_1, mask_black)
        # Bitwise-AND mask and original image
        return mask


    def detectSign(self):
        min_size_components = 550
        similitary_contour_with_circle = 0.65  # parameter
        count = 0
        current_sign = None
        coordinates = []
        file = open("Output.txt", "w")
        self.imageOrg = cv2.resize(self.imageOrg, (640, 480))
        print("Frame:{}".format(count))
        coordinate, ima, sign_type, text = self.localization(self.imageOrg, min_size_components,
                                                             similitary_contour_with_circle, model, count,
                                                             current_sign)
        cv2.imshow('Result', ima)
        out = cv2.imwrite('output.jpg', ima)
        plt.imshow(cv2.cvtColor(ima, cv2.COLOR_BGR2RGB))
        plt.show()