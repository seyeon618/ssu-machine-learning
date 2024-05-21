import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# 이미지를 전처리하는 함수
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# 품종을 예측하는 함수
def predict_breed(img_path):
    model = VGG16(weights='imagenet', include_top=True)
    img = preprocess_image(img_path)
    preds = model.predict(img)
    predicted_breed = np.argmax(preds[0])  # 가장 높은 확률을 가진 클래스의 인덱스
    return predicted_breed

def load_class_index():
    with open('imagenet_class_index.json') as f:
        class_index = json.load(f)
    # 클래스 인덱스를 품종으로 매핑
    class_index_to_breed = {int(idx): label for idx, (_, label) in class_index.items()}
    return class_index_to_breed

# 강아지 이미지 경로
img_path = 'data_set/dog2.jpg'

# 품종 예측 및 출력
predicted_breed = predict_breed(img_path)
class_index_to_breed = load_class_index()

# 예측된 품종 인덱스를 이용하여 품종 출력
if predicted_breed in class_index_to_breed:
    predicted_breed_name = class_index_to_breed[predicted_breed]
    print("Predicted Breed:", predicted_breed_name)
else:
    print("Invalid breed index:", predicted_breed)
