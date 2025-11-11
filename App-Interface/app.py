import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import io

# --- Configuration ---

# **CRITICAL**: Update this path to your actual saved model file!
MODEL_PATH = 'pathogen_model_final.tflite' 
IMAGE_SIZE = (224, 224) # Adjust this to the input size your model expects (e.g., (150, 150))
# **CRITICAL**: Update these labels to match the classes your model was trained on.
CLASS_LABELS = [
    'Healthy',
    'Fungi',
    'Bacteria',
    'Virus',
    'Pests'
]

# --- Model Loading and Caching ---

# @st.cache_resource caches the resource (the loaded model) so it only loads once, 
# speeding up subsequent interactions.
@st.cache_resource
def load_pathogen_model():
    """Loads the pre-trained Keras model from disk."""
    try:
        # Load the model from your specified path
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # Handle the case where the model file is missing or corrupted
        st.error(f"""
        **Model Loading Error:** Could not load the model from `{MODEL_PATH}`. 
        Please ensure your saved model file (e.g., `pathogen_model.h5`) 
        is in the same directory as `app.py` and the path is correct.

        If you are running this locally, the error details are: {e}
        """)
        return None

# --- Preprocessing Function ---

def preprocess_image(image_data):
    """
    Preprocesses the uploaded image to match the format the model expects.
    This typically involves resizing and normalization.
    """
    # Convert uploaded image to PIL Image object
    img = Image.open(image_data).convert('RGB')
    
    # Resize the image to the model's expected input size
    img = img.resize(IMAGE_SIZE)
    
    # Convert to NumPy array
    img_array = np.array(img)
    
    # Add an extra dimension for the batch size (Model expects shape: (1, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values (assuming your model was trained with normalized data [0, 1])
    img_array = img_array / 255.0
    
    return img_array

# --- Prediction and Display Logic ---

def predict_and_display(model, processed_image):
    """Runs the model prediction and displays the results."""
    st.subheader("Analysis Result")
    
    # 1. Run Prediction
    with st.spinner('Running Pathogen Analysis...'):
        # Simulate prediction time for a better user experience
        time.sleep(1) 
        
        predictions = model.predict(processed_image)
        # Squeeze the batch dimension (from (1, N_CLASSES) to (N_CLASSES,))
        probabilities = predictions[0]

    # 2. Find the highest probability class
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index] * 100
    predicted_label = CLASS_LABELS[predicted_index]

    # 3. Display the main result
    if "Healthy" in predicted_label:
        st.balloons()
        st.success(f"**Prediction: {predicted_label}** 🌿")
        st.info("Great news! The plant looks healthy.")
    else:
        st.error(f"**Prediction: {predicted_label}** 🚨")
        st.warning(f"High confidence ({confidence:.2f}%) of a pathogen detected. Please take action.")
    
    # 4. Display all probabilities (optional, but good for diagnostics)
    st.markdown("---")
    st.markdown("##### Detailed Probabilities")
    
    # Create a DataFrame for better display
    results_df = tf.convert_to_tensor(probabilities)
    
    # Streamlit uses the pandas library for st.bar_chart, which handles 
    # the conversion from NumPy/Tensor automatically.
    
    # Create a dictionary mapping labels to probabilities
    prob_dict = {label: prob * 100 for label, prob in zip(CLASS_LABELS, probabilities)}
    
    # Display results using a metric and bar chart
    cols = st.columns(len(CLASS_LABELS))
    for i, (label, prob) in enumerate(prob_dict.items()):
        cols[i].metric(label=label, value=f"{prob:.2f}%")
        
    st.bar_chart(prob_dict)

# --- Streamlit Main App Layout ---

def main():
    st.set_page_config(
        page_title="Plant Pathogen Identifier",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("🌿 Plant Pathogen Identifier")
    st.markdown(
        """
        Upload an image of a plant leaf to instantly classify it as 
        healthy or identify the specific pathogen affecting it.
        """
    )
    st.markdown("---")

    # Load the model
    model = load_pathogen_model()
    if model is None:
        # Exit if the model failed to load
        return 

    # File Uploader Widget
    uploaded_file = st.file_uploader(
        "Upload a plant leaf image (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess the image
            processed_image = preprocess_image(uploaded_file)
            
            # Run Prediction
            predict_and_display(model, processed_image)
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.warning("Please ensure the uploaded file is a valid image.")

if __name__ == "__main__":

    main()

