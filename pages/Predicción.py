import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps, ImageEnhance
from streamlit_drawable_canvas import st_canvas

# Cargar el modelo SVM desde el archivo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Crear la aplicación Streamlit
st.title("Reconocimiento de Dígitos MNIST con SVM 🖋️")
st.write("Dibuja un número en el canvas o sube una imagen para predecir.")

# Crear un canvas para dibujar el dígito
canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image):
    """ Preprocesa una imagen para hacerla compatible con el modelo SVM. """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Convertir a escala de grises
    image = image.convert("L")

    # Detectar si el fondo es más claro que el dígito
    mean_pixel = np.array(image).mean()
    if mean_pixel > 128:  # Si la imagen es mayormente blanca, invertir
        image = ImageOps.invert(image)


    # Redimensionar a 8x8
    image_resized = image.resize((8, 8), Image.Resampling.LANCZOS)

    # Convertir a array numpy y escalar los valores al rango de entrenamiento (0-16)
    image_array = (np.array(image_resized) / 255.0) * 16.0

    # Aplanar la imagen y aplicar el escalado
    image_flatten = image_array.flatten().reshape(1, -1)
    
    # Verificar si la imagen contiene contenido significativo antes de escalar
    if np.max(image_flatten) == 0:
        return None

    image_scaled = scaler.transform(image_flatten)

    return image_scaled


# Predicción de la imagen
def predict(image):
    """ Toma una imagen preprocesada y la clasifica usando el modelo SVM. """
    image_processed = preprocess_image(image)
    if image_processed is None:
        return None
    prediction = clf.predict(image_processed)
    return prediction[0]

# Predicción con la imagen del canvas
if st.button("Predecir desde Canvas"):
    if canvas.image_data is not None:
        img_array = np.array(canvas.image_data)[..., :3]  # Eliminar canal alpha
        
        # Verificar si el canvas tiene un dibujo
        if np.mean(img_array) < 250:  # Si la imagen no es completamente blanca
            prediction = predict(img_array)
            if prediction is not None:
                st.subheader("🧠 Predicción")
                st.success(f"El modelo predice que el número es: **{prediction}**")
            else:
                st.warning("⚠️ No se detectó un número válido en el canvas. Intenta dibujar de nuevo.")
        else:
            st.warning("⚠️ El canvas está vacío. Dibuja un número antes de predecir.")

# Subir una imagen de un dígito manuscrito
archivo_subido = st.file_uploader("📤 Sube una imagen de un dígito manuscrito (JPG o PNG)", type=["jpg", "png"])

if archivo_subido is not None:
    image = Image.open(archivo_subido)
    st.image(image, caption="📷 Imagen subida", width=150)

    prediction = predict(image)
    
    if prediction is not None:
        st.subheader("✅ Predicción")
        st.success(f"El modelo predice que el número es: **{prediction}**")
    else:
        st.error("❌ No se pudo procesar la imagen. Asegúrate de que contiene un dígito claro.")

st.write("🔍 **Esta app usa OpenCV y Scikit-learn para reconocer dígitos manuscritos.**")
