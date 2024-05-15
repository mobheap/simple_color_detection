import numpy as np
import cv2
from PIL import Image

yellow = [0, 255, 255] # BGR colorspace
red_bgr = [0, 0, 255]
blue_bgr = [255, 0, 0]
green_bgr = [0, 255, 0]

def main():
    # read webcam
    webcam = cv2.VideoCapture(0)

    # visualize webcam
    while True:
        ret, frame = webcam.read() # ret = read (binary) and frame = actual image
        
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lowerL, upperL = get_limits(yellow)
        mask = cv2.inRange(frameHSV, lowerL, upperL)
        
        mask_ = Image.fromarray(mask) # from np array (opencv) to pillow image
        bbox = mask_.getbbox() # bounding box
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4)
        
        cv2.imshow('Color Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

main()