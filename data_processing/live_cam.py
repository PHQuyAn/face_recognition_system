import cv2 as cv
from set_scale_cam import *

def live_cam():
    cap = cv.VideoCapture(0)
    make_1080p(cap)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv.imshow('frame', frame)
        key = cv.waitKey(20)

        if key == ord('q') or key == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    live_cam()
