# VisionWear
VisionWear is an AI-driven fashion recommendation system that suggests visually similar fashion items using deep learning–based feature extraction.
The system leverages a pretrained ResNet CNN model and KNN similarity search to provide accurate recommendations through an interactive Streamlit dashboard.

This project is developed for academic, internship, and demonstration purposes, showcasing real-world application of computer vision in fashion technology.

🎯 Key Features
👕 Browse fashion catalog
📸 Upload image-based recommendations
🧠 Deep learning feature extraction (ResNet)
🔍 Visual similarity using KNN
📊 Dataset analytics dashboard
🌐 Streamlit web interface
🧠 Technologies Used
Python
Streamlit
PyTorch
TorchVision
Scikit-learn
Pandas & NumPy
Pillow
⚙️ How It Works
Fashion images are processed using a pretrained ResNet18 CNN
Visual features are extracted from images
A KNN model finds visually similar items
Results are displayed via an interactive dashboard
🚀 How to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/surya323-ma/FashionIQ.git
cd FashionIQ
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Train the Model (One-Time)
bash
Copy code
python train_myntra_model.py
4️⃣ Run the Streamlit App
bash
Copy code
streamlit run fashion_app.py
🌐 Live Deployment
🔗 Streamlit App:
👉https://fashioniq.streamlit.app/

📂 Project Structure
Copy code
FashionIQ/
│
├── fashion_app.py
├── train_myntra_model.py
├── requirements.txt
├── fashion-recommendation.ipynb
├── models/
└── myntradataset/
📈 Use Cases
Fashion e-commerce recommendation systems

Visual product similarity search

AI-based fashion discovery platforms

Academic & internship demonstrations

👨‍💻 Developed By
Utkarsh Mishra
AI & Machine Learning Enthusiast

📜 Disclaimer
This project is intended for educational and demonstration purposes only.
The dataset used is publicly available and utilized to showcase recommendation techniques.
