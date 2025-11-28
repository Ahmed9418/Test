import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io

# --- Configuration ---
# Update this if your model is in a subfolder, e.g., 'model/pathogen_model.tflite'
MODEL_PATH = 'App-Interface/pathogen_model_final.tflite' 

CLASS_LABELS = [
    "Bacteria",
    "Fungi",
    "Healthy",
    "Pests",
    "Virus"
]

# --- 1. Model Loading ---
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model. Make sure '{MODEL_PATH}' is in the same directory.")
        st.error(str(e))
        return None

# --- 2. Robust Preprocessing ---
def preprocess_pil_image(pil_img, input_details):
    """
    Preprocesses the image for MobileNetV2:
    1. Fixes rotation (EXIF)
    2. Smart Crops to square (ImageOps.fit)
    3. Scales to [-1, 1]
    """
    input_shape = input_details[0]['shape'] # [1, 224, 224, 3]
    target_height = input_shape[1]
    target_width = input_shape[2]

    # A. Fix Rotation (Critical for phone photos)
    img = ImageOps.exif_transpose(pil_img)
    
    # B. Ensure RGB (removes Alpha channels)
    img = img.convert('RGB')
    
    # C. Smart Resize (Center Focus) - Prevents squashing
    img = ImageOps.fit(img, (target_width, target_height), Image.Resampling.LANCZOS)

    # D. Convert to Array & Scale
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # MobileNetV2 expects inputs in range [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

# --- 3. Prediction with TTA (Test Time Augmentation) ---
def predict_with_tta(interpreter, pil_image):
    """
    Runs prediction 3 times on the image (Original, Flipped, Zoomed)
    and averages the results. This fixes 'uncertain' predictions.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    
    # Pass 1: Original Image
    processed_1 = preprocess_pil_image(pil_image, input_details)
    interpreter.set_tensor(input_details[0]['index'], processed_1)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_details[0]['index'])[0])
    
    # Pass 2: Mirrored (Horizontal Flip)
    # Plants look the same flipped, but the model might catch features it missed
    flipped_img = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    processed_2 = preprocess_pil_image(flipped_img, input_details)
    interpreter.set_tensor(input_details[0]['index'], processed_2)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_details[0]['index'])[0])
    
    # Pass 3: Slight Internal Zoom (1.1x)
    # Crops 5% from edges to focus tighter
    width, height = pil_image.size
    crop_w = int(width * 0.05)
    crop_h = int(height * 0.05)
    zoomed_img = pil_image.crop((crop_w, crop_h, width - crop_w, height - crop_h))
    processed_3 = preprocess_pil_image(zoomed_img, input_details)
    interpreter.set_tensor(input_details[0]['index'], processed_3)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_details[0]['index'])[0])
    
    # Average the results
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

# --- 4. Main App UI ---
def main():
    st.set_page_config(page_title="Plant Care AI", page_icon="🌿")
    
    st.title("🌿 Plant Pathogen Identifier")
    st.markdown("Upload a photo of a plant leaf to detect diseases.")
    
    # Load Model
    interpreter = load_tflite_model()
    if interpreter is None:
        return

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load basic image
        original_image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(original_image, caption="Uploaded Image", width=300)
        
        # --- PREDICTION ---
        if st.button("Analyze Plant"):
            with st.spinner("Running AI Analysis..."):
                # Run the Robust TTA Prediction directly on the uploaded image
                probabilities = predict_with_tta(interpreter, original_image)
                
                # Get Top Prediction
                predicted_index = np.argmax(probabilities)
                confidence = probabilities[predicted_index] * 100
                predicted_label = CLASS_LABELS[predicted_index]
                
                # --- DISPLAY RESULTS ---
                st.markdown("---")
                st.subheader("Results")
                
                if "Healthy" in predicted_label:
                    st.balloons()
                    st.success(f"✅ Prediction: **{predicted_label}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.markdown("Your plant looks healthy! Keep up the good work.")
                else:
                    st.error(f"🚨 Pathogen Detected: **{predicted_label}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.warning("Immediate action may be required. Isolate this plant.")
                
                # Detailed Chart
                st.markdown("#### Confidence Breakdown")
                chart_data = {label: prob * 100 for label, prob in zip(CLASS_LABELS, probabilities)}
                st.bar_chart(chart_data)

if __name__ == "__main__":
    main()


