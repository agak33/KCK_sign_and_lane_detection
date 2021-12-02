import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import imutils
from PIL import Image as pilImage
from Image.image import Image
from keras.models import load_model
import warnings
from skimage import io, color

model = load_model(f'signDetection/Trafic_signs_model1.h5')
warnings.filterwarnings("ignore", category=DeprecationWarning)
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons',
           44: 'Other'}


class Detect(Image):
    def __init__(self, path: str, as_gray: bool = True):
        super().__init__(path)
        self.path = path


    # Preprocess image
    def constrastLimit(self, ima):
        img = color.rgb2hsv(ima)
        saturation = img[:, :, 1]
        q = np.percentile(saturation, 98)
        lower = np.array([0.0, q, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        ima = cv2.inRange(img, lower, upper)
        return ima


    def preprocess_image(self, ima):
        ima = self.constrastLimit(ima)
        return ima

    # Find Signs
    def removeSmallComponents(self, ima, threshold):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(ima, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        ima = np.zeros((output.shape), dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= threshold:
                ima[output == i + 1] = 255
        return ima


    def findContour(self, ima):
        cnts = cv2.findContours(ima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        return cnts


    def contourIsSign(self, perimeter, centroid, threshold):
        result = []
        for p in perimeter:
            p = p[0]
            distance = sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
            result.append(distance)
        max_value = max(result)
        signature = [float(dist) / max_value for dist in result]
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
            if is_sign and distance > max_distance and distance > distance_theshold:
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


    def classOfTypeSigne(self, sign_type):
        if sign_type > 0 and sign_type < 10:
            tekst = "ograniczenie predkosci"
        elif sign_type >= 10 and sign_type < 12:
            tekst = "zakaz jazdy"
        elif sign_type == 15:
            tekst = "stop"
        elif sign_type >= 20 and sign_type < 33:
            tekst = "znaki ostrzegawcze"
        elif sign_type >= 34 and sign_type < 41:
            tekst = "znaki nakazu"
        elif sign_type == 10:
            tekst = "zakaz wyprzedzania"
        elif sign_type == 13:
            tekst = "droga z pierwszenstwem"
        else:
            tekst = "inne"
        return tekst


    # classify type of sign
    def classify(self):
        ii = pilImage.open('0_.jpg')
        ii = ii.resize((30, 30))
        ii = np.expand_dims(ii, axis=0)
        ii = np.array(ii)
        predict_x = model.predict(ii)
        classes_x = np.argmax(predict_x, axis=1)
        return classes_x

    def localization(self, ima, min_size_components, similitary_contour_with_circle, count):
        original_image = ima.copy()
        binary_image = self.preprocess_image(ima)
        binary_image = self.removeSmallComponents(binary_image, min_size_components)
        contours = self.findContour(binary_image)
        sign, coordinate = self.findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
        text = ""
        sign_type = -1
        i = 0
        if sign is not None :
            cv2.imwrite(str(0) + '_' + text + '.jpg', sign)
            sign_type = int(self.classify()) + 1
            sign_type = sign_type if sign_type <= 43 else 44
            text = self.classOfTypeSigne(sign_type)
            cv2.imwrite(str(count) + '_' + text + '.jpg', sign)
        if sign_type > 0:
            cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 4)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] + 15), font, 1, (0, 0, 255), 2,
                        cv2.LINE_4)
        return coordinate, original_image, sign_type, text


    def detectSign(self,count):
        min_size_components = 550
        similitary_contour_with_circle = 0.65  # parameter
        #self.imageOrg = cv2.resize(self.imageOrg, (640, 480))
        print("Frame:{}".format(count))
        coordinate, ima, sign_type, text = self.localization(self.imageOrg, min_size_components,
                                                             similitary_contour_with_circle, count)
        self.finalImage = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
        #cv2.imshow('Result', ima )
        #out = cv2.imwrite(f'output{count}.jpg', ima)
