from LaneDetection.laneDetection import LaneDetection

if __name__ == '__main__':
    for i in range(66):
        img = LaneDetection(f'img/road{i}.jpg')
        img.findLane(f'results/road{i}.jpg')
