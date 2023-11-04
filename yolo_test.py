import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

'''
img/001520-20190925@102750-L2-S.jpg
img/001531-20190926@092020-R1-S.jpg
'''
if __name__ == "__main__":
    yolo = YOLO()
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
