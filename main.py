import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import openai
import os
import requests
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 이미지를 전처리하는 함수
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# 품종을 예측하는 함수
def predict_breed(img_path):
    model = MobileNetV2(weights='imagenet')
    img = preprocess_image(img_path)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=1)[0][0][1]
    return decoded_preds

# 강아지 이미지 경로
img_path1 = 'data_set/dog.jpg'
img_path2 = 'data_set/dog2.jpg'

# 품종 예측 및 출력
predicted_breed1 = predict_breed(img_path1)
predicted_breed2 = predict_breed(img_path2)

print("Predicted Breed 1:", predicted_breed1)
print("Predicted Breed 2:", predicted_breed2)

def generate_image(prompt):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=1,
    )
    image_url = response.data[0].url
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

prompt = f"Show an image of a puppy born from a crossbreed between a {predicted_breed1} and a {predicted_breed2}."
image_url = generate_image(prompt)
save_image(image_url, "generated_puppy.png")
