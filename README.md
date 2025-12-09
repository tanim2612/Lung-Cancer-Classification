# 🫁 Lung Cancer Classification Using Deep Learning

This project focuses on building a deep learning–based image classification model capable of identifying different types of lung cancer from histopathological images. It uses **PyTorch**, **CNN models (ResNet / Transfer Learning)**, and essential data preprocessing techniques to achieve robust classification performance.

---

## 📌 Project Overview

Lung cancer is one of the leading causes of cancer-related deaths globally. Early and accurate detection can significantly improve patient outcomes.  
In this project, we use **histopathological lung tissue images** and apply **deep learning techniques** to classify images into categories such as:

- **Benign**
- **Malignant**
- **Normal Lung Tissue**

The workflow includes:

1. Data loading and preprocessing  
2. Image augmentation using `torchvision.transforms`  
3. Model building with PyTorch (Transfer Learning)  
4. Training & validation  
5. Evaluation using accuracy, confusion matrix, and classification report  

---

## 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- PIL
- tqdm

---

## 📁 Project Structure

```
Lung-Cancer-Classification/
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   └── best_model.pth
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── main.ipynb
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Lung-Cancer-Classification.git
cd Lung-Cancer-Classification
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

Make sure you have the `requirements.txt` file, then run:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### **Run the training script**

```bash
python src/train.py
```

### **Evaluate the model**

```bash
python src/evaluate.py
```

### **Or open the notebook**

```bash
jupyter notebook main.ipynb
```

---

## 📊 Results

The model evaluation includes:

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

Visualizations are generated for:

- Loss curves
- Accuracy curves
- Confusion matrix heatmap

---

## 📌 Future Improvements

- Experiment with EfficientNet / Vision Transformers  
- Add Grad-CAM for visual explanations  
- Deploy model using Flask or Streamlit  
- Improve dataset balancing  

---

## 🤝 Contributing

Contributions are welcome!  
If you’d like to enhance this project, feel free to submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Acknowledgments

Special thanks to the open datasets and researchers who contributed to histopathological image collections used for academic research.

