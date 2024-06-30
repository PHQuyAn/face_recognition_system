import cv2 as cv
import os
import sys

def cropImage(video_name):
    video_file = os.path.join("videos",video_name + ".mp4")

    #Create folder to save images
    output_dir = os.path.join("images", video_name)
    if os.path.exists(output_dir):
       return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    print(video_file)

    cap = cv.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error when open video")
        exit()

    frame = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_name = f"img{frame_count}.jpg"
        image_path = os.path.join(output_dir, image_name)

        cv.imwrite(image_path, frame)

        frame_count +=1

        key = cv.waitKey(20)
        if key == ord('q') or key==27:
            break

    cap.release()
    cv.destroyAllWindows()

    #Remove .mp4 extentions
    for folder_name in os.listdir('images'):
        if folder_name.endswith(".mp4"):
            new_folder_name = folder_name[:-4]
            os.rename(os.path.join('images',folder_name),
                      os.path.join('images',new_folder_name))

def remove_mp4_extensions():
    for folder_name in os.listdir('images'):
        if folder_name.endswith(".mp4"):
            new_folder_name = folder_name[:-4]
            os.rename(os.path.join('images',folder_name),
                      os.path.join('images',new_folder_name))

def main():
    for video_name in os.listdir('videos'):
        print(video_name)
        if video_name.endswith('.mp4'):
            video_nm = video_name[:-4]
            cropImage(video_nm)
    #remove_mp4_extensions()

if __name__ == "__main__":
    main()