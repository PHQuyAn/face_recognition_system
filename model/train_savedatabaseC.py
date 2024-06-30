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
        face = img[y:y + h, x:x + w]
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


def create_table_with_labels(labels):
    conn = psycopg2.connect(
        dbname="ceh_face",
        user="postgres",
        password="123",
        host="localhost"
    )
    cur = conn.cursor()

    # Ensure labels are unique and properly formatted for SQL
    labels = [label.replace(" ", "_").replace("-", "_") for label in labels]
    columns = ", ".join([f'{label} vector(512)' for label in labels])

    print(columns)
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS user_embeddings (
        id SERIAL PRIMARY KEY,
        {columns}
    );
    """

    print(create_table_query)
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()


def save_embeddings_to_db(embeddings, labels):
    conn = psycopg2.connect(
        dbname='ceh_face',
        user='postgres',
        password='123',
        host='localhost'
    )
    cursor = conn.cursor()

    # Create dictionary to store embeddings by label
    embeddings_dict = {label.replace(" ", "_").replace("-", "_"): [] for label in set(labels)}
    for embedding, label in zip(embeddings, labels):
        label = label.replace(" ", "_").replace("-", "_")
        embeddings_dict[label].append(embedding)

    # Prepare rows for insertion
    rows = []
    max_len = max(len(v) for v in embeddings_dict.values())
    for i in range(max_len):
        row = []
        for label in embeddings_dict.keys():
            if i < len(embeddings_dict[label]):
                row.append(embeddings_dict[label][i].tolist())
            else:
                row.append([None] * 512)  # Fill with null if no embedding at this position
        rows.append(row)

    # Insert rows into table
    for row in rows:
        row_str = ", ".join(
            [f"ARRAY{str(embedding)}::vector" if embedding[0] is not None else "NULL" for embedding in row])
        insert_query = f"""
        INSERT INTO user_embeddings ({", ".join([f'"{label.lower()}"' for label in embeddings_dict.keys()])})
        VALUES ({row_str});
        """
        cursor.execute(insert_query)
        conn.commit()

    cursor.close()
    conn.close()


def main():
    # #img = cv.imread('./../data_processing/data_images/Quy_An/iamg9.jpg')
    # # part1(img)
    # # part2(img)
    #
    # # Load face and save face
    # faceloading = FACELOADING("./../data_processing/data_images")
    # X, Y = faceloading.load_classes()
    # faceloading.plot_images()
    #
    # # FaceNet part
    # EMBEDDED_X = []
    #
    # for img in X:
    #     EMBEDDED_X.append(get_embedding(img))
    #
    # EMBEDDED_X = np.asarray(EMBEDDED_X)
    # np.savez_compressed('faces_embeddings_done_4classes.npz',
    #                     EMBEDDED_X, Y)

    # Load data from file faces_embeddings_done_4classes.npz
    data = np.load('faces_embeddings_done_4classes.npz')
    EMBEDDED_X = data['arr_0']
    Y = data['arr_1']

    unique_labels = set(Y)
    print(unique_labels)
    create_table_with_labels(unique_labels)

    save_embeddings_to_db(EMBEDDED_X, Y)


if __name__ == "__main__":
    main()
