# A Novel Deep Learning Model for Early Detection and Progression Prediction of Diabetic Retinopathy

## 📌 Project Overview
This project focuses on the **early detection** and **progression prediction** of **Diabetic Retinopathy (DR)** using advanced deep learning models. It explores transformer-based architectures and integrates predictions with a lightweight IoT device for real-world application.

## 🎯 Objectives
- Detect DR in early stages to prevent vision loss.
- Accurately classify DR into five stages.
- Compare performance of traditional CNNs with transformer-based models.
- Integrate prediction output with a working frontend interface.

## 🧠 Models Used
- **Vision Transformer (ViT)**
- **Swin Transformer**
- **Quantum Vision Transformer (QViT)**
- **CNNs**: ResNet-50, EfficientNet-B4

## 🧪 Dataset
- **APTOS 2019 Blindness Detection Dataset**
  - Fundus images: 3,662 training + 1,928 testing
  - DR Classes: 0 (No DR) to 4 (Proliferative DR)  
  - [Kaggle Dataset Link](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

## ⚙️ Technologies Used
- Python, PyTorch  
- HTML, CSS, JavaScript (for frontend)  
- Kaggle GPU (for training)  

## 📊 Results
- **Swin Transformer** achieved the best performance in terms of accuracy and AUC.
- **QViT** showed promising results with hybrid quantum features.
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC

## 🔮 Future Scope
- Collect more real-world labeled images.
- Add explainable AI features like attention maps.
- Deploy lightweight models on IoT devices for rural clinics.
