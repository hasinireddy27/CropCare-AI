# 🌿 CropCare AI – Intelligent Crop Disease Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras"/>
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-green?style=for-the-badge"/>
</p>

## 📖 Overview

**CropCare AI** is an AI-powered crop disease detection web application that helps farmers and agriculture enthusiasts identify plant diseases from leaf images.

The system uses **Transfer Learning with MobileNetV2** to classify crop diseases from the **PlantVillage dataset** and provides predictions with confidence scores through an easy-to-use web interface.

The application is fully responsive and works seamlessly on both **desktop and mobile devices**.

---

## ✨ Features

* 🌱 Detect crop diseases from leaf images
* 📷 Upload images or capture directly from a mobile camera
* 🤖 AI-powered disease prediction using MobileNetV2
* 📊 Displays confidence score
* 📈 Shows Top-3 predicted diseases
* 🌿 Detects healthy and diseased leaves
* 📱 Responsive UI for desktop and mobile
* ⚡ Fast prediction using TensorFlow & Flask

---

## 🧠 AI Model

* **Model:** MobileNetV2 (Transfer Learning)
* **Framework:** TensorFlow 2.21
* **Image Size:** 224 × 224
* **Classes:** 38 Crop Disease Categories
* **Dataset:** PlantVillage Dataset

---

## 🛠️ Technology Stack

| Category        | Technologies            |
| --------------- | ----------------------- |
| Programming     | Python                  |
| Backend         | Flask                   |
| Deep Learning   | TensorFlow, Keras       |
| AI Model        | MobileNetV2             |
| Frontend        | HTML5, CSS3, JavaScript |
| Version Control | Git & GitHub            |

---

## 📂 Project Structure

```text
CropCareAI/
│
├── app.py
├── requirements.txt
├── services/
├── templates/
├── static/
├── model/
├── training/
└── uploads/
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/hasinireddy27/CropCare-AI.git
```

### Navigate to the project

```bash
cd CropCare-AI
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📸 Application Workflow

```text
Leaf Image
      │
      ▼
Image Upload
      │
      ▼
Image Preprocessing
      │
      ▼
MobileNetV2 Prediction
      │
      ▼
Disease Classification
      │
      ▼
Confidence Score
      │
      ▼
Prediction Result
```

---

## 📊 Future Enhancements

* 🌍 Multi-language Support
* 🎤 Voice-Based Disease Explanation
* 📱 Progressive Web App (PWA)
* ☁️ Cloud Deployment
* 🤖 AI-powered Treatment Recommendations
* 📈 Prediction History Dashboard

---

## 👩‍💻 Developer

**Hasini Reddy Namireddy**

* 🎓 B.Tech – Artificial Intelligence & Machine Learning
* 💻 AI | Machine Learning | Computer Vision | Generative AI Enthusiast

---

## ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub. It helps support the project and motivates future development.
