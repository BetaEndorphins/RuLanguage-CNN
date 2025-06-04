from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json


model = load_model('model_1.keras')

with open('class_names.json', 'r', encoding='utf-8') as f:
    class_indices = json.load(f)

# Обратный словарь: индексы -> названия классов
index_to_class = {v: k for k, v in class_indices.items()}

# Путь к изображению для теста
img_path = '/Users/graygrape/Desktop/IMG_3837_1.jpg'  # замени на нужное изображение

# Предобработка изображения
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0)

# Предсказание
pred = model.predict(img_array)
pred_index = np.argmax(pred)

print(f'Модель предсказала: {index_to_class[pred_index]} (достоверность: {pred[0][pred_index]:.2f})')
