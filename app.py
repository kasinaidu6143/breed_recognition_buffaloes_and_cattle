import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import numpy as np
import gradio as gr
import json
import google.generativeai as genai

# --- 1. LOAD ASSETS ---
MODEL_PATH = "indian_cattle_breed_recognizer.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 224

# Set up Gemini API Key (Using an environment variable or direct input)
# Ensure you have set the GOOGLE_API_KEY environment variable.
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAhy0HXH6ZP1IUfBMWun07ttGadTWOJcVs" # Replace with your actual key if needed

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the class names from your training
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

# --- 2. DYNAMIC INFO FETCHING FUNCTION ---
def get_breed_details(breed_name):
    """Fetches summary from Gemini dynamically based on the predicted breed."""
    try:
        # Using Gemini 2.5 Flash for reliable text generation
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"Provide a concise, 3-4 sentence summary about the '{breed_name}' indigenous cattle/buffalo breed from India. Focus on its origin, characteristics, and significance."
        
        response = model.generate_content(prompt)
        
        if response.text:
            return response.text
        else:
            return f"Could not generate details for '{breed_name}'. This breed is part of the 74 indigenous varieties identified in India."
            
    except Exception as e:
        return f"Error fetching details from Gemini API: {str(e)}\n\n(Note: Please ensure you have set a valid GOOGLE_API_KEY)"

# --- 3. CORE LOGIC ---
def predict_and_fetch_info(input_img):
    if input_img is None:
        return {"None": 0.0}, "Please upload an image first."
        
    # Preprocess
    img = tf.image.resize(input_img, [IMG_SIZE, IMG_SIZE])
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    # Model Prediction
    predictions = model.predict(img_array)[0]
    top_index = np.argmax(predictions)
    predicted_breed = class_names[top_index]
    confidence = float(predictions[top_index])

    # Dynamic Wikipedia Fetching
    wiki_info = get_breed_details(predicted_breed)
    
    # Return chart data and the text summary

    
    # Return chart data and the markdown text
    top_3_results = {class_names[i]: float(predictions[i]) for i in np.argsort(predictions)[-3:]}
    return top_3_results, wiki_info

# --- 4. GRADIO INTERFACE ---
custom_theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    spacing_size="lg",
    text_size="lg",
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
    block_shadow="none",
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
    button_primary_text_color="#ffffff",
    panel_background_fill="transparent",
    input_background_fill="transparent",
)

css_code = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    background-color: #f8fafc !important;
    background-image: 
        radial-gradient(at 10% 20%, hsla(250,100%,74%,0.15) 0px, transparent 50%),
        radial-gradient(at 90% 10%, hsla(190,100%,56%,0.15) 0px, transparent 50%),
        radial-gradient(at 50% 80%, hsla(330,100%,76%,0.15) 0px, transparent 50%) !important;
    background-attachment: fixed !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #0f172a !important;
}

/* Header Styling */
.app-header {
    text-align: center;
    margin: 20px auto 40px auto;
    padding: 40px 20px;
    background: rgba(255, 255, 255, 0.6) !important;
    border-radius: 32px !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,1) !important;
    max-width: 900px;
}

.app-title {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px !important;
    line-height: 1.2 !important;
    letter-spacing: -0.03em !important;
}

.app-subtitle {
    font-size: 1.25rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    margin: 0 !important;
}

/* Glass Columns */
.col-glass {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 32px !important;
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,1) !important;
    padding: 30px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 20px !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.col-glass:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 30px 50px -10px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,1) !important;
}

/* Image Upload Styling */
.image-upload {
    border-radius: 20px !important;
    overflow: hidden !important;
    background: rgba(255,255,255,0.4) !important;
    border: 2px dashed rgba(148, 163, 184, 0.5) !important;
}
.image-upload:hover {
    border-color: #8b5cf6 !important;
}

/* Button Styling (Enhanced) */
.analyze-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.25rem !important;
    padding: 20px !important;
    border-radius: 24px !important;
    box-shadow: 0 12px 24px -10px rgba(124, 58, 237, 0.5) !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
    margin-top: 10px !important;
}
.analyze-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 20px 32px -10px rgba(124, 58, 237, 0.6) !important;
}
.analyze-btn:active {
    transform: scale(0.98) !important;
}

/* Results Formatting */
.results-label {
    background: rgba(255, 255, 255, 0.5) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.7) !important;
}
.results-label .gr-label-value {
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    color: #1e293b !important;
}
.results-label .gr-progress-bar {
    background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%) !important;
    border-radius: 10px !important;
}

/* Textarea / Information text */
.info-text textarea {
    font-size: 1.15rem !important;
    line-height: 1.7 !important;
    color: #334155 !important;
    background: rgba(255, 255, 255, 0.5) !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
    font-weight: 500 !important;
}

/* Mobile Responsiveness */
.main-row {
    gap: 30px !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

@media (max-width: 768px) {
    .main-row {
        flex-direction: column !important;
        gap: 20px !important;
    }
    .app-title {
        font-size: 2.2rem !important;
    }
    .app-subtitle {
        font-size: 1.1rem !important;
    }
    .col-glass {
        padding: 20px !important;
        border-radius: 24px !important;
    }
    .app-header {
        padding: 30px 20px !important;
        margin-bottom: 20px !important;
        border-radius: 24px !important;
    }
}

footer { display: none !important; }
"""

with gr.Blocks(theme=custom_theme, css=css_code) as demo:
    gr.HTML("""
    <div class="app-header">
        <h1 class="app-title">🐄 Indian Livestock Intelligence</h1>
        <p class="app-subtitle">Identify indigenous cattle varieties and fetch real-time data from the web instantly.</p>
    </div>
    """)
    
    with gr.Row(elem_classes="main-row"):
        with gr.Column(scale=1, min_width=320, elem_classes="col-glass"):
            input_image = gr.Image(label="Upload Cow/Buffalo Photo", type="numpy", elem_classes="image-upload")
            btn = gr.Button("✨ Analyze Breed ✨", variant="primary", size="lg", elem_classes="analyze-btn")
            
        with gr.Column(scale=1, min_width=320, elem_classes="col-glass"):
            output_label = gr.Label(num_top_classes=3, label="Detection Probabilities", elem_classes="results-label")
            output_markdown = gr.Textbox(label="Breed Information", interactive=False, lines=6, max_lines=8, elem_classes="info-text")

    btn.click(
        fn=lambda: gr.update(interactive=False), 
        outputs=btn
    ).then(
        fn=predict_and_fetch_info, 
        inputs=input_image, 
        outputs=[output_label, output_markdown]
    ).then(
        fn=lambda: gr.update(interactive=True), 
        outputs=btn
    )

if __name__ == "__main__":
    demo.launch(share=True)
