from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json

# Параметры
img_width, img_height = 224, 224
batch_size = 32
epochs = 30

# Аугментация данных
datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2,
    rotation_range = 20,
    width_shift_range = 0.1,
    height_shift_range = 0.1, #!
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_data = datagen.flow_from_directory(
    'images',
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training',
    shuffle = True,
    seed = 42
)

val_data = datagen.flow_from_directory(
    'images',
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = True,
    seed = 42
)

# Модель
base_model = MobileNetV2(
    weights = 'imagenet',
    include_top = False,
    input_shape = (img_width, img_height, 3)
)
base_model.trainable = False # Заморозка весов

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation = 'relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation = 'softmax')
])

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True,
    verbose = 1
)

# Обучение
history = model.fit(
    train_data,
    validation_data = val_data,
    epochs = epochs,
    callbacks = [early_stop]
)

model.save('model_1.keras')

with open('class_names.json', 'w', encoding='utf-8') as f:
    json.dump(train_data.class_indices, f, ensure_ascii=False)

print("Модель успешно обучена и сохранена!")