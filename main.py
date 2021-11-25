from LaneDetection.laneDetection import LaneDetection

if __name__ == '__main__':
    # img = Image(f'img/road1.jpg')
    # img.houghTransform()
    for i in range(1, 15):
        img = LaneDetection(f'img/road{i}.jpg')
        img.findLane()
        #img.show()


