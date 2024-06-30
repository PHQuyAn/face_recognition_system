import cv2 as cv
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import psycopg2

import json

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

embedder = FaceNet()

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

    def plot_images(self):
        plt.figure(figsize=(16, 12))
        for num, image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y) // ncols + 1
            plt.subplot(nrows, ncols, num + 1)
            plt.imshow(image)
            plt.axis('off')

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]


def save_embeddings_to_db(embeddings, labels):
    conn = psycopg2.connect(
        dbname='ceh_face',
        user='postgres',
        password='123',
        host='localhost'
    )
    cursor = conn.cursor()

    for embedding, label in zip(embeddings, labels):
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'  # Chuyển đổi mảng numpy thành chuỗi
        cursor.execute("INSERT INTO embeddings (label, embedding) VALUES (%s, %s::vector(512)) RETURNING id", (label, embedding_str))
        inserted_id = cursor.fetchone()[0]
        print(f"Inserted ID: {inserted_id}")
        conn.commit()

    cursor.close()
    conn.close()

def main():
    # Đọc dữ liệu từ file faces_embeddings_done_4classes.npz
    data = np.load('faces_embeddings_done_4classes.npz')
    EMBEDDED_X = data['arr_0']
    Y = data['arr_1']

    save_embeddings_to_db(EMBEDDED_X, Y)

if __name__ == "__main__":
    main()
