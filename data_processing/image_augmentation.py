import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io

import cv2 as cv
import random
import numpy as np
import os
from PIL import Image

def random_brightness(img):
  # Tạo ảnh đen hoàn toàn
  zeros = np.zeros(img.shape, img.dtype)

  # Tạo giá trị alpha ngẫu nhiên trong phạm vi [0.5, 2.0]
  alpha = random.uniform(0.5, 2.0)

  # Kết hợp ảnh gốc và ảnh đen với tỷ lệ trọng số ngẫu nhiên
  out = cv.addWeighted(img, alpha, zeros, 1-alpha, 0)

  return out

def random_noise(img):
    # Generate random noise
    noise = np.random.normal(loc = 0, scale=random.uniform(0, 30), size = img.shape)

    # Add noise to img
    out = img + noise
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def augmentate_Img():
    datagen = ImageDataGenerator(
                rotation_range=50,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range= 0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='reflect'
        # Try fill mode: nearest, constant, reflect, wrap
            )

    for folder_name in os.listdir('images'):
        save_folder = os.path.join("augmented",folder_name)

        if os.path.exists(save_folder):
            continue
        else:
            print(save_folder, ': created new folder')
            os.makedirs(save_folder, exist_ok=True)

        img_directory = os.path.join('images',folder_name + '/')
        print(img_directory)
        #SIZE = 128
        dataset = []
        # image dataset for each folder
        my_images = os.listdir(img_directory)
        for i, image_name in enumerate(my_images):
            if (image_name.split('.')[1] == 'jpg'):
                x = io.imread(img_directory + image_name)
                # Brighten the image by 20%

                x = random_brightness(x)
                x = random_noise(x)
                # image = Image.fromarray(image, 'RGB')
                # image = image.resize((SIZE,SIZE))
                # dataset.append(np.array(image))
                # x = io.imread('images/Quy_An/img0.jpg')

                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in datagen.flow(x, batch_size = 16,
                                          save_to_dir = save_folder,
                                          save_prefix = 'aug',
                                          save_format = 'jpg'):
                    i +=1
                    if i>2:
                        break

def main():
    augmentate_Img()

if __name__=="__main__":
    main()

# x = io.imread('images/Quy_An/img0.jpg')
#
# x = x.reshape((1,) + x.shape)
# i = 0
# for batch in datagen.flow(x, batch_size = 16,
#                           save_to_dir = 'augmented',
#                           save_prefix = 'aug',
#                           save_format = 'jpg'):
#     i +=1
#     if i>20:
#         break



                #x = np.array(dataset)
                #print(x.shape)

#     for batch in datagen.flow(x, batch_size = 16,
#                               save_to_dir = 'augmented',
#                               save_prefix = 'aug',
#                               save_format = 'jpg'):
#         i+=1
#         if i>20:
#             break

# i = 0
# for batch in datagen.flow_from_directory(directory='images',
#                                          batch_size=32,
#                                          target_size=(480,640),
#                                          save_to_dir="augmented",
#                                          save_prefix="aug",
#                                          save_format="jpg"):
#     i+=1
#     if i>31:
#         break
