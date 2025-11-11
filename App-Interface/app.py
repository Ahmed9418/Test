import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os

# --- Configuration ---

# TFLite model file name
MODEL_PATH = 'App-Interface/pathogen_model_final.tflite'
# MobileNetV2 standard input size
IMAGE_SIZE = (224, 224) 

# **CRITICAL**: Update these labels to match the classes your model was trained on.
CLASS_LABELS = [
    'Healthy',
    'Fungi',
    'Bacteria',
    'Virus',
    'pests'
]

# --- Model Loading and Caching ---

# Use cache to load the TFLite interpreter just once
@st.cache_resource
def load_tflite_model():
    """Loads the TFLite interpreter from disk."""
    
    # Check if the file exists first (crucial for Streamlit Cloud deployment)
    if not os.path.exists(MODEL_PATH):
        st.error(f"""
        **Model Not Found:** Could not find the model file at `{MODEL_PATH}`. 
        Please ensure your `pathogen_model.tflite` file is correctly uploaded 
        in the `App-Interface` folder on GitHub.
        """)
        return None

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        st.success("✅ TFLite Model loaded successfully!")
        return interpreter
    except Exception as e:
        st.error(f"""
        **Model Loading Error:** Failed to load TFLite Interpreter. 
        Error details: {e}
        """)
        return None

# --- Preprocessing Function ---

def preprocess_image(image_data, input_details):
    """
    Preprocesses the uploaded image to match the format the TFLite model expects.
    """
    # Get the input shape expected by the model
    input_shape = input_details[0]['shape'] # e.g., (1, 224, 224, 3)
    
    # Convert uploaded image to PIL Image object
    img = Image.open(image_data).convert('RGB')
    
    # Resize the image to the model's expected input size (index 1 and 2 of shape)
    img = img.resize((input_shape[1], input_shape[2]))
    
    # Convert to NumPy array
    img_array = np.array(img, dtype=np.float32) # TFLite often uses float32
    
    # Add an extra dimension for the batch size
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values (assuming normalization to [0, 1] for MobileNetV2)
    img_array = img_array / 255.0
    
    return img_array

# --- Prediction and Display Logic ---

def predict_and_display(interpreter, processed_image):
    """Runs the TFLite model prediction and displays the results."""
    st.subheader("Analysis Result")
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 1. Run Prediction
    with st.spinner('Running Pathogen Analysis...'):
        # Set the tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Invoke the model (run prediction)
        interpreter.invoke()
        
        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        probabilities = predictions[0]

    # 2. Find the highest probability class
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index] * 100
    predicted_label = CLASS_LABELS[predicted_index]

    # 3. Display the main result
    st.markdown(f"#### Predicted Label: **{predicted_label}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")
    
    if "Healthy" in predicted_label:
        st.balloons()
        st.success("🌿 Great news! The plant is classified as healthy.")
    else:
        st.error("🚨 Pathogen Detected! Immediate action may be required.")
        st.warning(f"Detected Pathogen: **{predicted_label}**.")
    
    # 4. Display all probabilities (good for diagnostics)
    st.markdown("---")
    st.markdown("##### Detailed Probabilities")
    
    # Create a dictionary mapping labels to probabilities
    prob_dict = {label: prob * 100 for label, prob in zip(CLASS_LABELS, probabilities)}
    
    # Display results using a bar chart for comparison
    st.markdown("Distribution of Classification Scores:")
    st.bar_chart(prob_dict)

# --- Streamlit Main App Layout ---

def main():
    st.set_page_config(
        page_title="Plant Pathogen Identifier",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("🌿 Plant Pathogen Identifier (TFLite)")
    st.markdown(
        """
        Upload an image of a plant leaf to instantly classify it as 
        healthy or identify the specific pathogen affecting it.
        ---
        """
    )

    # Load the interpreter
    interpreter = load_tflite_model()
    if interpreter is None:
        # Exit if the model failed to load or wasn't found
        return 
        
    # Get model input details once for preprocessing
    input_details = interpreter.get_input_details()

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
            processed_image = preprocess_image(uploaded_file, input_details)
            
            # Run Prediction
            predict_and_display(interpreter, processed_image)
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.warning("Please ensure the uploaded file is a valid image.")

if __name__ == "__main__":
    main()

