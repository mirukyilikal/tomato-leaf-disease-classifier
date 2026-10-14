# Tomato Leaf Disease Classifier

A Streamlit demo for a fine-tuned EfficientNetB0 model that classifies tomato-leaf images into 10 PlantVillage classes.

## Model results

- Fine-tuned test accuracy: **95.26%**
- Fine-tuned test loss: **0.1344**
- Test set size: **1,816 images**
- Architecture: EfficientNetB0 transfer learning
- Input size: 224 × 224 RGB

These results are from the held-out PlantVillage test split. They do not guarantee equivalent performance on real-world field images.

## 1. Project structure

```text
tomato_leaf_streamlit/
├── app.py
├── class_names.json
├── requirements.txt
├── README.md
└── models/
    └── tomato_finetuned_best.keras   # Add this file
```

## 2. Add your trained model

In Google Colab, confirm that this file exists:

```text
/content/tomato-disease-models/tomato_finetuned_best.keras
```

Download it from Colab's Files panel, then place it in the `models` folder in this project. The final path must be:

```text
models/tomato_finetuned_best.keras
```

Do not rename it differently unless you also update `MODEL_PATH` in `app.py`.

## 3. Run locally on Windows

Open a terminal in this project directory. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Then open the local URL Streamlit prints, usually `http://localhost:8501`.

TensorFlow compatibility depends on your Python version and operating system. If installation fails, use a Python version supported by the installed TensorFlow release (Python 3.11 is a practical choice for this project).

## 4. Use the app

1. Start the app.
2. Upload a JPG, JPEG, PNG, or WEBP tomato-leaf image.
3. Review the predicted class, confidence, and top three predictions.

The app keeps pixel values in the 0–255 range because the saved EfficientNetB0 model includes its own input rescaling. Do not add `/255` normalization unless you change the model's preprocessing.

## 5. Important limitations

- This is an educational classifier, not a professional plant-health diagnosis.
- Confidence is not a guarantee of correctness.
- The test score is dataset-specific; field images may differ in lighting, backgrounds, camera quality, and disease appearance.
- The model can confuse visually similar classes. Compare predictions with local agricultural guidance when making crop-management decisions.

## 6. GitHub note

Model files can be large. Before pushing the model to GitHub, check the repository's file-size limits and your chosen hosting platform's rules. You can keep the model in a separate download location or use Git LFS if appropriate.
