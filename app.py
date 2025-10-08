import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# === Cargar modelo entrenado ===
model = tf.keras.models.load_model("modelo_residuos_finetuned.h5")

# === Clases detectadas ===
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# === Función de predicción ===
def predict_image(image: Image.Image):
    # Redimensionar y convertir a array
    img = image.resize((224, 224))
    img = np.array(img)

    # Preprocesamiento igual que en entrenamiento
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = model.predict(img)
    return class_labels[np.argmax(pred)], np.max(pred)

# === Interfaz Streamlit ===
st.title("♻️ Clasificador de Residuos con IA")
st.write("Sube una imagen de un residuo y el modelo intentará clasificarla.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen subida
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Predicción
    label, confidence = predict_image(image)
    st.markdown(f"### 🔍 Predicción: **{label}**")
    st.write(f"Confianza: {confidence*100:.2f}%")
    