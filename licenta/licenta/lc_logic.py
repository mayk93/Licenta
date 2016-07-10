import cv2
import numpy
from matplotlib import pyplot


GRAYSCALE = cv2.IMREAD_GRAYSCALE  # Convert to grays cale
COLORED = cv2.IMREAD_COLOR  # Remove alpha but keep colours
NORMAL = cv2.IMREAD_UNCHANGED  # Do not change image, keep colours and alpha


class ImageProcessor(object):
    def __init__(self, image_path):
        self.image_path = image_path
        # Read as gray scale to make processing simpler
        self.image = cv2.imread(self.image_path, GRAYSCALE)

    def __del__(self):
        pass

    def test(self):
        '''
        Silly test method that displays the image server side - Has no practical use
        :return:
        '''
        cv2.imwrite('/Users/Michael/Desktop/gscat.jpg', self.image)


def process(file_path):
    image_processor = ImageProcessor(file_path)
    image_processor.test()
    return {"path": file_path}