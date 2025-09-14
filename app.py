import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- PAGE CONFIGURATION (Customizes the tab and layout) ---
st.set_page_config(
    page_title="Beach Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Injects custom styles) ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# You can create a style.css file for more complex styles, or just inject simple ones
st.markdown("""
<style>
.big-font {
    font-size:2rem !important;
    font-weight: bold;
    color: #28a745;
}
.confidence-font {
    font-size:1.2rem !important;
    color: #6c757d;
}
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
MODEL_PATH = 'garbage_classifier.keras'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- IMPORTANT ---
# You must update this list to match the class order from your training
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'] #<-- UPDATE THIS!


# --- IMAGE PREPROCESSING ---
def preprocess_image(image_data):
    image = Image.open(image_data).resize((180, 180))
    image_array = np.array(image)
    if image_array.shape[2] == 4: # Handle PNG transparency
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- SIDEBAR CONTENT ---
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a Convolutional Neural Network (CNN) to classify images of beach waste. "
    "Upload an image, and the model will predict its material type."
)
st.sidebar.success("Model ready and loaded successfully!")

# --- MAIN PAGE LAYOUT ---
st.title("♻️ Beach Waste Classification App")
st.write("---")

uploaded_file = st.file_uploader(
    "Upload an image of waste to classify...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    with col2:
        st.write("### Classifying...")
        with st.spinner('Please wait...'):
            try:
                processed_image = preprocess_image(uploaded_file)
                prediction = model.predict(processed_image)
                score = tf.nn.softmax(prediction[0])

                predicted_class = CLASS_NAMES[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.write("### Prediction Result:")
                st.markdown(f'<p class="big-font">{predicted_class.capitalize()}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="confidence-font">Confidence: {confidence:.2f}%</p>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")