import cv2 as cv
import os

def video_capture(nameFile, FPS = 25):
    cam = cv.VideoCapture(0)
    cc = cv.VideoWriter_fourcc(*'XVID')

    # Create the "videos" directory if it doesn't exist
    videos_dir = "videos"
    os.makedirs(videos_dir, exist_ok=True)  # Ensures directory creation

    # Combine filename with directory path
    nameFile = os.path.join(videos_dir, nameFile + '.mp4')
    file = cv.VideoWriter(nameFile, cc, FPS, (640,480))

    if not cam.isOpened():
        print("error opening camera")
        exit()

    total_frame = FPS * 15
    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()

        # if frame is read correctly ret is True
        if not ret:
            print("error in retrieving frame")
            break

        cv.imshow('frame', frame)
        file.write(frame)

        key = cv.waitKey(20)
        if key == ord('q') or key == 27:
            break

        frame_count += 1
        if frame_count>=total_frame:
            break

    cam.release()
    file.release()
    cv.destroyAllWindows()

def main():
    name = input("Your name: ")
    video_capture(name, 25)

if __name__=="__main__":
    main()

