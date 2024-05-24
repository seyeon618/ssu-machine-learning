import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import openai
import os
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
openai.api_key = os.getenv('API_KEY')

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

def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url

def save_image(image_url, file_path):
    # 이미지 다운로드 및 저장
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(image_response.content)
        print(f"Image saved to {file_path}")
    else:
        print("Failed to retrieve the image")

# 강아지 이미지 경로
img_path1 = 'data_set/dog.jpg'
img_path2 = 'data_set/dog2.jpg'

# 품종 예측 및 출력
predicted_breed1 = predict_breed(img_path1)
predicted_breed2 = predict_breed(img_path2)
class_index_to_breed = load_class_index()

dog1 = ''
dog2 = ''

# 예측된 품종 인덱스를 이용하여 품종 출력 (첫번째 강아지)
if predicted_breed1 in class_index_to_breed:
    dog1 = class_index_to_breed[predicted_breed1]
    print("Predicted Breed:", dog1)
else:
    print("Invalid breed index:", predicted_breed1)

# 예측된 품종 인덱스를 이용하여 품종 출력 (두번째 강아지)
if predicted_breed2 in class_index_to_breed:
    dog2 = class_index_to_breed[predicted_breed2]
    print("Predicted Breed:", dog2)
else:
    print("Invalid breed index:", predicted_breed2)

prompt = f"Show an image of a puppy born from a crossbreed between a {dog1} and a {dog2}."
image_url = generate_image(prompt)
save_image(image_url, "generated_puppy.png")
