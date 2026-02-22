# 🌿 Plant Disease Detection using Deep Learning

This project detects diseases in plant leaves using a Convolutional Neural Network (CNN) model trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease). It includes a Google Colab training notebook and a Streamlit web interface for predictions.

---

## 📌 Features

- Detects plant diseases from images
- Trained on thousands of labeled leaf images
- CNN model built using TensorFlow/Keras
- Deployed via a Streamlit web app

---

## 🛠 Tech Stack

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- Streamlit (for deployment)
- Google Colab (for training)

---

## 🚀 Getting Started

### 🔧 Step 1: Train the Model on Google Colab

1. Open the Colab notebook:  
   📓 [`model_training_notebook/Plant_Disease_Prediction_with_CNN.ipynb`](model_training_notebook/Plant_Disease_Prediction_with_CNN.ipynb)

2. Set Colab to use GPU:  
   - Go to **Runtime > Change runtime type**
   - Set **Hardware accelerator** to **GPU**

3. Upload your Kaggle API key:
   - Go to [https://www.kaggle.com](https://www.kaggle.com)
   - Click your profile picture → **Account**
   - Scroll to the **API** section
   - Click **Create New API Token** → it downloads `kaggle.json`
   - Upload `kaggle.json` to Colab
   - Upload `test_images` to Colab

4. Run all cells in the notebook to train the CNN model.

5. After training:

   * Download the trained model file (e.g., `model.h5`)
   * Save the `class_indices.json` generated during training

---

### 💻 Step 2: Set Up the Project Locally

1. Clone this repo:

   ```bash
   git clone https://github.com/CraigDaGama/Plant-Disease-Prediction-with-CNN
   cd Plant-Disease-Prediction-with-CNN
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Replace the following files with your trained versions:

   * `trained_model/model.h5` – your trained model
   * `trained_model/class_indices.json` – class labels mapping

---

### ▶️ Step 3: Run the Web App

```bash
streamlit run app/main.py
```

---

## 📁 Project Structure

```
plant-disease-detection/
├── app/
│   └── main.py                  # Streamlit web app
├── trained_model/
│   └── model.h5                 # Trained model (replace this)
├── requirements.txt
├── Dockerfile
├── class_indices.json       # Class indices (replace this)
└── README.md
```



## 📜 License

This project is licensed under the **MIT License**.
