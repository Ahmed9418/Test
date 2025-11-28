import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw
import io
import os

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'pathogen_model_final.tflite')

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
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. Robust Preprocessing ---
def preprocess_pil_image(pil_img, input_details):
    input_shape = input_details[0]['shape'] 
    target_height = input_shape[1]
    target_width = input_shape[2]

    # Convert to RGB
    img = pil_img.convert('RGB')
    
    # Resize to 224x224 (Model Input Size)
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert to Array & Scale to [-1, 1]
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

# --- 3. Prediction with TTA ---
def predict_with_tta(interpreter, pil_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    
    # Pass 1: Original
    processed_1 = preprocess_pil_image(pil_image, input_details)
    interpreter.set_tensor(input_details[0]['index'], processed_1)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_details[0]['index'])[0])
    
    # Pass 2: Mirrored (Horizontal Flip)
    flipped_img = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    processed_2 = preprocess_pil_image(flipped_img, input_details)
    interpreter.set_tensor(input_details[0]['index'], processed_2)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_details[0]['index'])[0])
    
    return np.mean(predictions, axis=0)

# --- 4. Main App UI ---
def main():
    st.set_page_config(page_title="Plant Care AI", page_icon="🌿")
    st.title("🌿 Plant Pathogen Identifier")
    
    interpreter = load_tflite_model()
    if interpreter is None: return

    uploaded_file = st.file_uploader("Upload Plant Photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and Fix Rotation immediately
        original_image = Image.open(uploaded_file)
        original_image = ImageOps.exif_transpose(original_image)
        
        # --- TARGETING TOOL ---
        st.markdown("### 🎯 Step 1: Target the Disease")
        st.info("Use the controls to place the RED BOX over the sickest leaf.")
        
        # 1. Controls
        col_controls_1, col_controls_2 = st.columns(2)
        with col_controls_1:
            zoom = st.slider("🔍 Zoom", 1.0, 5.0, 1.0, 0.1)
        
        # Calculate Box Size (Force Square Crop)
        img_w, img_h = original_image.size
        min_dim = min(img_w, img_h)
        box_size = int(min_dim / zoom)
        
        # Calculate Max Offsets
        max_x = img_w - box_size
        max_y = img_h - box_size
        
        with col_controls_2:
            # Use percentages for sliders to make them responsive
            x_pct = st.slider("↔️ Move Left/Right", 0, 100, 50)
            y_pct = st.slider("↕️ Move Up/Down", 0, 100, 50)
            
        # Convert % to pixels
        x_offset = int((x_pct / 100) * max_x)
        y_offset = int((y_pct / 100) * max_y)
        
        # Define Crop Box
        left = x_offset
        top = y_offset
        right = x_offset + box_size
        bottom = y_offset + box_size
        
        # Perform Crop
        final_image = original_image.crop((left, top, right, bottom))
        
        # --- VISUALIZATION ---
        # Draw Red Box on a Copy of Original
        preview_img = original_image.copy()
        draw = ImageDraw.Draw(preview_img)
        draw.rectangle([left, top, right, bottom], outline="red", width=int(min_dim*0.02))
        
        # Display Side-by-Side
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Full Image (Red Box = Analysis Area)")
            st.image(preview_img, use_column_width=True)
        with col2:
            st.caption("What the AI Sees (Must show disease!)")
            st.image(final_image, use_column_width=True)
            
        # Warning if resolution is too low
        if box_size < 224:
            st.warning("⚠️ Warning: You zoomed in too much! The image is blurry. The AI might struggle.")

        # --- ANALYZE ---
        st.markdown("---")
        if st.button("Analyze Target Area", type="primary"):
            with st.spinner("Analyzing..."):
                probabilities = predict_with_tta(interpreter, final_image)
                
                # Results
                pred_idx = np.argmax(probabilities)
                confidence = probabilities[pred_idx] * 100
                label = CLASS_LABELS[pred_idx]
                
                if "Healthy" in label:
                    st.success(f"✅ Prediction: **{label}** ({confidence:.1f}%)")
                else:
                    st.error(f"🚨 Pathogen: **{label}** ({confidence:.1f}%)")
                
                st.bar_chart({l: p*100 for l, p in zip(CLASS_LABELS, probabilities)})

if __name__ == "__main__":
    main()
