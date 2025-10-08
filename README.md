[README.md](https://github.com/user-attachments/files/22778892/README.md)
# DetectorDeBasura

# ♻️ Clasificación de Residuos con MobileNetV2

## 🎯 Objetivo
Desarrollar un sistema de inteligencia artificial capaz de clasificar imágenes de residuos en seis categorías: **cartón, vidrio, metal, papel, plástico y basura general**, utilizando técnicas de visión artificial y aprendizaje profundo.

---

## 📂 Dataset
- **Fuente**: [TrashNet Dataset](https://github.com/garythung/trashnet)  
- **Número de clases**: 6 (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`)  
- **Número de imágenes**: ~2,500  
- **Estructura de carpetas**:  
  - `trashnet/dataset-resized/cardboard/`  
  - `trashnet/dataset-resized/glass/`  
  - `trashnet/dataset-resized/metal/`  
  - `trashnet/dataset-resized/paper/`  
  - `trashnet/dataset-resized/plastic/`  
  - `trashnet/dataset-resized/trash/`  
- **División**: 80% entrenamiento, 20% validación  

---

## ⚙️ Preprocesamiento
- Redimensionamiento de imágenes a **224x224 píxeles**  
- Normalización con `preprocess_input` de MobileNetV2  
- Aumentación de datos: rotaciones, flips horizontales, zoom  
- División en entrenamiento y validación con `ImageDataGenerator`  

---

## 🧠 Modelo
- **Arquitectura base**: MobileNetV2 preentrenado en ImageNet  
- **Capas añadidas**:  
  - GlobalAveragePooling2D  
  - Dense(256, ReLU)  
  - Dropout(0.4)  
  - Dense(6, Softmax)  
- **Fine‑tuning**: se descongelaron las últimas 50 capas de MobileNetV2  
- **Optimizador**: Adam con learning rate = 1e‑5  
- **Pérdida**: Categorical Crossentropy  
- **Épocas**: hasta 25 (con EarlyStopping)  

---

## 📊 Resultados
- **Precisión en validación**: 80%  
- **Reporte de métricas**:
                 precision    recall  f1-score   support
   cardboard       0.98      0.74      0.84        80
       glass       0.75      0.83      0.79       100
       metal       0.78      0.78      0.78        82
       paper       0.84      0.92      0.88       118
     plastic       0.73      0.74      0.74        96
       trash       0.48      0.44      0.46        27
  
- **Matriz de confusión**:
- <img width="800" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/c0862fa3-18c9-437b-8628-7d67c9b1e3e7" />
 
---

## 🌐 Interfaz
Se implementó una interfaz con **Streamlit** que permite:  
- Subir una imagen desde el navegador  
- Procesarla con el modelo entrenado  
- Mostrar la clase predicha y el nivel de confianza  

Ejecutar con:
```bash
streamlit run app.py
```

---

## 📦 Requisitos
Instalar dependencias con:
```bash
pip install -r requirements.txt
```

Ejemplo de `requirements.txt`:
- tensorflow  
- keras  
- opencv-python  
- scikit-learn  
- matplotlib  
- seaborn  
- streamlit  
- numpy  
- pandas  

---

## 📌 Limitaciones
- Dataset relativamente pequeño → riesgo de sobreajuste  
- Clases visualmente similares (ej. vidrio vs plástico) pueden confundirse  
- Sensible a iluminación y calidad de la imagen  

---

## 🚀 Conclusiones
- Se logró entrenar un modelo eficiente con MobileNetV2 para clasificación de residuos  
- El sistema se integró en una aplicación sencilla con Streamlit  
- Futuro trabajo: ampliar dataset, probar arquitecturas más robustas y optimizar para dispositivos móviles  

---

## 👨‍💻 Autor
- **Nombre**: Alex Hernandez  
- **Universidad Rafael Urdaneta**  
- **Carrera**: Ingeniería de Computación – Inteligencia Artificial  

---

👉 Ahora sí, si lo pegas así en GitHub, se verá como un README normal con títulos, listas y secciones, sin que todo quede dentro de un cuadro.  

¿Quieres que te prepare también un **requirements.txt con versiones exactas** (ej. TensorFlow 2.15, Streamlit 1.39, etc.) para que tu profesor pueda instalar sin problemas?
