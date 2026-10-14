from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "tomato_finetuned_best.keras"
CLASS_NAMES_PATH = APP_DIR / "class_names.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide",
)

CLASS_NAMES_FALLBACK = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]


def display_class_name(name: str) -> str:
    """Convert internal dataset labels into readable labels."""
    name = name.replace("Tomato___", "")
    name = name.replace("_", " ")
    if name == "Spider mites Two-spotted spider mite":
        return "Spider mites (two-spotted spider mite)"
    if name == "Tomato Yellow Leaf Curl Virus":
        return "Tomato Yellow Leaf Curl Virus"
    return name


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            names = json.load(file)
        if len(names) != 10:
            raise ValueError("class_names.json must contain exactly 10 class labels.")
        return names
    return CLASS_NAMES_FALLBACK


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to the model's 224x224 input and keep pixel values in [0, 255]."""
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    # Do not divide by 255: this EfficientNetB0 model includes its own rescaling.
    return np.expand_dims(array, axis=0)


def main():
    st.title("🍅 Tomato Leaf Disease Classifier")
    st.markdown(
        "Upload a tomato-leaf image to get a model prediction across **10 classes**."
    )

    with st.sidebar:
        st.header("About this model")
        st.write("Architecture: EfficientNetB0 transfer learning")
        st.write("Input size: 224 × 224 pixels")
        st.write("Test accuracy: **95.26%** on the held-out PlantVillage test set")
        st.caption(
            "Reported test accuracy is dataset-specific and is not a guarantee "
            "of performance on field photographs."
        )

    if not MODEL_PATH.exists():
        st.error("The trained model file was not found.")
        st.markdown(
            "Place your fine-tuned model at "
            "`models/tomato_finetuned_best.keras` inside this project folder."
        )
        st.code(
            "tomato_leaf_streamlit/\n"
            "├── app.py\n"
            "├── class_names.json\n"
            "├── requirements.txt\n"
            "└── models/\n"
            "    └── tomato_finetuned_best.keras"
        )
        st.stop()

    try:
        model = load_model(str(MODEL_PATH))
        class_names = load_class_names()
    except Exception as exc:
        st.error(f"Could not load the model or class labels: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a tomato-leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image showing a tomato leaf.",
    )

    if uploaded_file is None:
        st.info("Upload an image to start a prediction.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("This file could not be opened as an image. Try another image.")
        st.stop()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing the image..."):
        try:
            batch = preprocess_image(image)
            probabilities = model.predict(batch, verbose=0)[0]
            if len(probabilities) != len(class_names):
                st.error(
                    f"The model returned {len(probabilities)} outputs, "
                    f"but {len(class_names)} labels are configured."
                )
                st.stop()
            top_indices = np.argsort(probabilities)[::-1][:3]
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    best_idx = int(top_indices[0])
    best_label = display_class_name(class_names[best_idx])
    confidence = float(probabilities[best_idx])

    with right:
        st.subheader("Prediction")
        st.metric("Predicted class", best_label)
        st.metric("Model confidence", f"{confidence:.1%}")
        st.progress(float(np.clip(confidence, 0.0, 1.0)))

        if confidence >= 0.80:
            st.success("The model strongly favors this class among its 10 labels.")
        elif confidence >= 0.50:
            st.warning("The prediction is moderately confident. Review the alternatives.")
        else:
            st.warning("The prediction is uncertain. Consider a clearer image.")

        st.subheader("Top 3 predictions")
        for rank, idx in enumerate(top_indices, start=1):
            label = display_class_name(class_names[int(idx)])
            score = float(probabilities[int(idx)])
            st.write(f"**{rank}. {label}** — {score:.1%}")
            st.progress(float(np.clip(score, 0.0, 1.0)))

    with st.expander("How to interpret this result"):
        st.write(
            "The displayed confidence is the model's predicted probability for a "
            "class, not a calibrated guarantee that the class is correct."
        )
        st.write(
            "This is an educational machine-learning demonstration, not a "
            "professional plant-health diagnosis. If a crop is affected, compare "
            "the result with local agricultural guidance or a qualified agronomist."
        )

    st.caption(
        "Model trained on the tomato subset of PlantVillage. Real-world performance "
        "may differ due to lighting, camera quality, backgrounds, and field conditions."
    )


if __name__ == "__main__":
    main()
