import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

# === Generadores de datos ===
data_dir = "trashnet/dataset-resized"
xz 
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Modelo base ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Fine-tuning: descongelamos últimas 50 capas
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# === Construcción del modelo ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# === Compilación ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# === Entrenamiento ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=[es]
)

# === Guardar modelo ===
model.save("modelo_residuos_finetuned.h5")