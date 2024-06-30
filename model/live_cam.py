import cv2 as cv
import os
import tensorflow as tf
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_distances
import joblib
from ultralytics import YOLO

# Giảm mức độ cảnh báo của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khởi tạo mô hình FaceNet và tải dữ liệu
facenet = FaceNet()
faces_embedding = np.load("faces_embeddings_done_4classes.npz")
EMBEDDED_X = faces_embedding['arr_0']

Y = faces_embedding['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Khởi tạo mô hình YOLO để phát hiện khuôn mặt
yolo_model = YOLO("yolov8l_100e.pt")

# Khởi tạo mô hình SVM để nhận diện khuôn mặt
svm_model = joblib.load('svm_model_160x160.pkl')

# Khởi tạo webcam
cap = cv.VideoCapture(0)

def is_unknown_face(embedding, known_embeddings, threshold=0.5):
    distances = cosine_distances(embedding, known_embeddings)
    min_distance = np.min(distances)
    return min_distance > threshold, min_distance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Dùng mô hình YOLO để phát hiện khuôn mặt
    results = yolo_model(rgb_img)
    boxes = results[0].boxes

    for box in boxes:
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_right_x = int(box.xyxy.tolist()[0][2])
        bottom_right_y = int(box.xyxy.tolist()[0][3])

        cv.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50, 200, 120), 2)

        img = rgb_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)

        is_unknown, min_distance = is_unknown_face(ypred, EMBEDDED_X, threshold=0.2)
        if is_unknown:
            final_name = "Unknown"
            accuracy = None
        else:
            face_name = svm_model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)[0]  # Lấy phần tử đầu tiên
            accuracy = (1 - min_distance) * 100

        cv.putText(frame, str(final_name), (top_left_x, top_left_y - 30), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)  # Giảm độ dày của chữ xuống 2
        if accuracy is not None:
            cv.putText(frame, f"Accuracy: {accuracy:.2f}%", (top_left_x, top_left_y - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow("Output", frame)
    keys = cv.waitKey(1)
    if keys == ord('q') or keys == 27:
        break

cap.release()
cv.destroyAllWindows()
